from collections import defaultdict

import torch
from torch.utils.data import Dataset


def build_hard_banks(
    embeddings, labels, indices, negative_bank_size=32, device=None, chunk_size=256
):
    """Build split-local hardest positive and nearest cross-class negatives."""
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    subset = embeddings[indices].to(device=device, dtype=torch.float32)
    labels = labels[indices].cpu()
    hardest_positive = {}
    negative_bank = {}
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        similarities = (subset[start:stop] @ subset.T).cpu()
        rows = torch.arange(stop - start)
        similarities[rows, torch.arange(start, stop)] = -torch.inf
        for offset, local_index in enumerate(range(start, stop)):
            global_index = int(indices[local_index])
            same = labels == labels[local_index]
            same[local_index] = False
            positive_candidates = torch.nonzero(same).flatten()
            if len(positive_candidates) == 0:
                raise ValueError("Mỗi lớp cần ít nhất hai ảnh trong split")
            hardest = positive_candidates[
                torch.argmin(similarities[offset, positive_candidates])
            ]
            hardest_positive[global_index] = int(indices[hardest])
            candidates = torch.nonzero(labels != labels[local_index]).flatten()
            count = min(negative_bank_size, len(candidates))
            nearest = candidates[
                torch.argsort(
                    similarities[offset, candidates], descending=True, stable=True
                )[:count]
            ]
            negative_bank[global_index] = indices[nearest].clone()
    return hardest_positive, negative_bank


def generate_pair_indices(
    labels,
    indices,
    hardest_positive,
    negative_bank,
    seed,
    epoch,
):
    """Generate two positives and two negatives per anchor deterministically."""
    labels = labels.cpu()
    indices = indices.cpu()
    by_class = defaultdict(list)
    for index in indices.tolist():
        by_class[int(labels[index])].append(index)
    generator = torch.Generator().manual_seed(seed + epoch)
    anchors, partners, targets = [], [], []
    shuffled = indices[torch.randperm(len(indices), generator=generator)]
    for anchor in shuffled.tolist():
        same = [value for value in by_class[int(labels[anchor])] if value != anchor]
        random_positive = same[
            torch.randint(len(same), (1,), generator=generator).item()
        ]
        hard_positive = hardest_positive[anchor]
        positives = [hard_positive, random_positive]
        if positives[0] == positives[1] and len(same) > 1:
            positives[1] = same[(same.index(positives[1]) + 1) % len(same)]

        bank = negative_bank[anchor]
        hard_negative = int(
            bank[torch.randint(len(bank), (1,), generator=generator).item()]
        )
        other_classes = [
            label for label in sorted(by_class) if label != int(labels[anchor])
        ]
        chosen_class = other_classes[
            torch.randint(len(other_classes), (1,), generator=generator).item()
        ]
        candidates = by_class[chosen_class]
        random_negative = candidates[
            torch.randint(len(candidates), (1,), generator=generator).item()
        ]
        for partner in positives:
            anchors.append(anchor)
            partners.append(partner)
            targets.append(1.0)
        for partner in (hard_negative, random_negative):
            anchors.append(anchor)
            partners.append(partner)
            targets.append(0.0)
    return (
        torch.tensor(anchors, dtype=torch.long),
        torch.tensor(partners, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32),
    )


class PairIndexDataset(Dataset):
    def __init__(self, embeddings, anchors, partners, targets, bidirectional=True):
        if not (len(anchors) == len(partners) == len(targets)):
            raise ValueError("Pair indices và targets không cùng độ dài")
        self.embeddings = embeddings
        if bidirectional:
            self.anchors = torch.cat((anchors, partners))
            self.partners = torch.cat((partners, anchors))
            self.targets = torch.cat((targets, targets))
        else:
            self.anchors = anchors
            self.partners = partners
            self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (
            self.embeddings[self.anchors[index]],
            self.embeddings[self.partners[index]],
            self.targets[index],
        )
