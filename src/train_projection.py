import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Sampler


class ProxyAnchorLoss(nn.Module):
    """Device-agnostic Proxy Anchor loss for local labels [0, num_classes)."""

    def __init__(self, num_classes, embedding_dim, margin=0.1, alpha=32.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.alpha = alpha
        self.proxies = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    @staticmethod
    def _log1p_sum_exp(values):
        zero = values.new_zeros(1)
        return torch.logsumexp(torch.cat((zero, values)), dim=0)

    def forward(self, embeddings, labels):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Embedding shape không phù hợp Proxy Anchor")
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("Labels không khớp embeddings")
        if labels.numel() == 0:
            raise ValueError("Proxy Anchor không nhận batch rỗng")
        if labels.min().item() < 0 or labels.max().item() >= self.num_classes:
            raise ValueError("Labels nằm ngoài khoảng proxy")

        cosine = F.linear(
            F.normalize(embeddings, dim=1), F.normalize(self.proxies, dim=1)
        )
        positive_terms = []
        negative_terms = []
        for class_index in range(self.num_classes):
            positive_mask = labels == class_index
            if positive_mask.any():
                positive_logits = -self.alpha * (
                    cosine[positive_mask, class_index] - self.margin
                )
                positive_terms.append(self._log1p_sum_exp(positive_logits))
            negative_logits = self.alpha * (
                cosine[~positive_mask, class_index] + self.margin
            )
            negative_terms.append(self._log1p_sum_exp(negative_logits))
        return torch.stack(positive_terms).mean() + torch.stack(negative_terms).mean()


class BalancedBatchSampler(Sampler):
    """Yield deterministic class-balanced batches, sampling images as needed."""

    def __init__(
        self, labels, classes_per_batch, samples_per_class, seed=42, num_batches=None
    ):
        self.labels = torch.as_tensor(labels, dtype=torch.long).cpu()
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.epoch = 0
        by_class = defaultdict(list)
        for index, label in enumerate(self.labels.tolist()):
            by_class[label].append(index)
        self.by_class = {label: torch.tensor(indices) for label, indices in by_class.items()}
        self.classes = sorted(self.by_class)
        if classes_per_batch > len(self.classes):
            raise ValueError("classes_per_batch vượt số lớp hiện có")
        batch_size = classes_per_batch * samples_per_class
        self.num_batches = num_batches or math.ceil(len(self.labels) / batch_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        for _ in range(self.num_batches):
            class_order = torch.randperm(len(self.classes), generator=generator)
            selected_classes = [
                self.classes[index]
                for index in class_order[:self.classes_per_batch].tolist()
            ]
            batch = []
            for label in selected_classes:
                candidates = self.by_class[label]
                if len(candidates) >= self.samples_per_class:
                    chosen = torch.randperm(len(candidates), generator=generator)[
                        :self.samples_per_class
                    ]
                else:
                    chosen = torch.randint(
                        len(candidates),
                        (self.samples_per_class,),
                        generator=generator,
                    )
                batch.extend(candidates[chosen].tolist())
            yield batch


def stratified_train_validation_split(labels, validation_fraction, seed):
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    validation_indices = []
    for label in sorted(labels.unique().tolist()):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        indices = indices[torch.randperm(len(indices), generator=generator)]
        validation_count = max(1, round(len(indices) * validation_fraction))
        if validation_count >= len(indices):
            raise ValueError(f"Lớp {label} không đủ mẫu để chia train/validation")
        validation_indices.extend(indices[:validation_count].tolist())
        train_indices.extend(indices[validation_count:].tolist())
    return torch.tensor(train_indices), torch.tensor(validation_indices)


def select_best_epoch(history):
    if not history:
        raise ValueError("Training history rỗng")
    best = min(
        history,
        key=lambda row: (
            -row["validation_recall@1"],
            row["validation_loss"],
            row["epoch"],
        ),
    )
    return int(best["epoch"])
