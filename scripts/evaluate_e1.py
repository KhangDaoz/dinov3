import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidential import binary_ranking_metrics, risk_coverage_metrics
from retrieval import (
    cosine_top_indices,
    recall_from_indices,
    rerank_top_n_by_uncertainty,
)
from train_evidential import e1_paths, load_cls_cache
from utils import (
    atomic_json_save,
    atomic_torch_save,
    load_config,
    load_json,
    resolve_device,
    sha256_file,
)


def _optional_mean(values):
    return values.mean().item() if values.numel() else None


def _oracle_rerank(indices, labels, top_n):
    result = indices.clone()
    prefix = result[:, :top_n]
    positives = labels[prefix].eq(labels[:, None])
    order = torch.argsort(
        (~positives).to(torch.int8), dim=1, descending=False, stable=True
    )
    result[:, :top_n] = torch.gather(prefix, 1, order)
    return result


def run_e1_evaluation(config_path):
    config = load_config(config_path)
    if config.get("experiment") != "e1":
        raise ValueError("evaluate_e1 yêu cầu experiment='e1'")
    paths = e1_paths(config["output_dir"])
    required = [
        paths["checkpoint"], paths["history"], paths["validation"],
        paths["test_uncertainty"], paths["metrics"], paths["manifest"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu E1 artifacts: {', '.join(missing)}")

    manifest = load_json(paths["manifest"])
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = paths["root"] / name
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"E1 artifact checksum không khớp: {name}")
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    test_payload = torch.load(
        paths["test_uncertainty"], map_location="cpu", weights_only=True
    )
    train_embeddings, test_embeddings, labels_payload, source_hashes = (
        load_cls_cache(config)
    )
    del train_embeddings
    if source_hashes != checkpoint.get("source_cache_sha256"):
        raise ValueError("Checkpoint không khớp CLS source cache")

    uncertainty = test_payload.get("uncertainty")
    labels = labels_payload["test"]["labels"].long()
    if uncertainty.shape != labels.shape or not torch.isfinite(uncertainty).all():
        raise ValueError("Test uncertainty không hợp lệ")
    if torch.any((uncertainty <= 0) | (uncertainty > 1)).item():
        raise ValueError("Test uncertainty phải nằm trong (0, 1]")
    if test_payload.get("paths") != labels_payload["test"]["paths"]:
        raise ValueError("Test uncertainty paths lệch CLS cache")

    device = resolve_device(config["device"])
    started = time.perf_counter()
    selected_top_n = int(checkpoint["selected_top_n"])
    retained = max(selected_top_n, max(config["recall_k"]))
    baseline_indices = cosine_top_indices(
        test_embeddings,
        retained,
        config["retrieval_chunk_size"],
        device=device,
    )
    reranked_indices = rerank_top_n_by_uncertainty(
        baseline_indices, uncertainty, selected_top_n
    )
    baseline_recall = recall_from_indices(
        baseline_indices, labels, config["recall_k"]
    )
    reranked_recall = recall_from_indices(
        reranked_indices, labels, config["recall_k"]
    )
    source_recall = load_json(
        Path(config["source_output_dir"]) / "metrics.json"
    ).get("evaluation", {}).get("metrics", {})
    for name, value in baseline_recall.items():
        if name not in source_recall or abs(value - source_recall[name]) > 1e-6:
            raise RuntimeError(f"E1-A không tái tạo CLS baseline tại {name}")

    baseline_correct = labels[baseline_indices[:, 0]].eq(labels)
    errors = ~baseline_correct
    detection = binary_ranking_metrics(uncertainty, errors)
    risk_coverage = risk_coverage_metrics(uncertainty, errors)
    correct_uncertainty = uncertainty[baseline_correct]
    incorrect_uncertainty = uncertainty[errors]
    diagnostics = {
        **detection,
        **risk_coverage,
        "mean_uncertainty": uncertainty.mean().item(),
        "mean_uncertainty_top1_correct": _optional_mean(correct_uncertainty),
        "mean_uncertainty_top1_incorrect": _optional_mean(incorrect_uncertainty),
        "top1_errors": int(errors.sum()),
    }

    oracle_indices = _oracle_rerank(baseline_indices, labels, selected_top_n)
    oracle_recall = recall_from_indices(
        oracle_indices, labels, config["recall_k"]
    )
    atomic_torch_save(
        {
            "selected_top_n": selected_top_n,
            "baseline_top_indices": baseline_indices,
            "reranked_top_indices": reranked_indices,
        },
        paths["reranking"],
    )
    metrics = load_json(paths["metrics"])
    metrics.update(
        {
            "status": "evaluated",
            "evaluation": {
                "e1_a_cls_cosine": baseline_recall,
                "e1_b_uncertainty_reranking": reranked_recall,
                "oracle_top_n": oracle_recall,
                "selected_top_n": selected_top_n,
                "uncertainty_diagnostics": diagnostics,
                "num_queries": len(labels),
                "num_gallery": len(labels),
                "self_exclusion": True,
                "cosine_tie_policy": "stable_gallery_index_ascending",
                "uncertainty_tie_policy": "preserve_cosine_order",
                "device": str(device),
                "seconds": time.perf_counter() - started,
            },
        }
    )
    atomic_json_save(metrics, paths["metrics"])
    manifest["status"] = "evaluated"
    manifest["artifact_sha256"]["metrics.json"] = sha256_file(paths["metrics"])
    manifest["artifact_sha256"]["reranking_scores.pt"] = sha256_file(
        paths["reranking"]
    )
    atomic_json_save(manifest, paths["manifest"])
    return metrics["evaluation"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate E1 uncertainty reranking.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e1.yaml")
    )
    args = parser.parse_args()
    result = run_e1_evaluation(args.config)
    print(result)


if __name__ == "__main__":
    main()
