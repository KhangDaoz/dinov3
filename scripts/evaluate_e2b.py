import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pair_confidence import PairConfidenceNetwork
from retrieval import (
    cosine_top_candidates,
    fuse_candidate_rankings,
    recall_from_indices,
)
from train_evidential import load_cls_cache
from train_pair_confidence import e2b_paths, score_candidates
from utils import (
    atomic_json_save,
    atomic_torch_save,
    load_config,
    load_json,
    resolve_device,
    sha256_file,
)


def load_model(checkpoint, device):
    architecture = checkpoint["architecture"]
    model = PairConfidenceNetwork(
        embedding_dim=768,
        hidden_dims=architecture["hidden_dims"],
        dropout=architecture["dropout"],
        feature_mode=architecture["feature_mode"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def run_e2b_evaluation(config_path):
    config = load_config(config_path)
    if config.get("experiment") != "e2b":
        raise ValueError("evaluate_e2b yêu cầu experiment='e2b'")
    paths = e2b_paths(config["output_dir"])
    required = [paths["checkpoint"], paths["metrics"], paths["manifest"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu E2B artifacts: {', '.join(missing)}")
    e1_metrics_path = Path(config["e1_output_dir"]) / "metrics.json"
    if not e1_metrics_path.exists():
        raise FileNotFoundError("E2B cần E1 đã chạy và có metrics.json")
    e1_metrics = load_json(e1_metrics_path)
    if e1_metrics.get("status") != "evaluated":
        raise ValueError("E1 phải ở trạng thái evaluated")

    manifest = load_json(paths["manifest"])
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = paths["root"] / name
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"E2B artifact checksum không khớp: {name}")
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    _, embeddings, labels_payload, source_hashes = load_cls_cache(config)
    if checkpoint["source_cache_sha256"] != source_hashes:
        raise ValueError("E2B checkpoint không khớp CLS cache")
    labels = labels_payload["test"]["labels"].long()
    paths_list = labels_payload["test"]["paths"]
    device = resolve_device(config["device"])
    model = load_model(checkpoint, device)
    started = time.perf_counter()

    max_candidates = max(config["retrieval"]["candidate_top_n_grid"])
    indices, cosine = cosine_top_candidates(
        embeddings,
        max_candidates,
        config["retrieval_chunk_size"],
        device,
    )
    confidence = score_candidates(model, embeddings, indices, config, device)
    candidate_top_n = int(checkpoint["candidate_top_n"])
    weight = float(checkpoint["lambda"])
    b2_indices, _ = fuse_candidate_rankings(
        indices, cosine, confidence, 0.0, candidate_top_n
    )
    b3_indices, b3_scores = fuse_candidate_rankings(
        indices, cosine, confidence, weight, candidate_top_n
    )
    b4_indices, _ = fuse_candidate_rankings(
        indices, cosine, confidence, 1.0, candidate_top_n
    )
    b0 = recall_from_indices(indices, labels, config["recall_k"])
    b2 = recall_from_indices(b2_indices, labels, config["recall_k"])
    b3 = recall_from_indices(b3_indices, labels, config["recall_k"])
    b4 = recall_from_indices(b4_indices, labels, config["recall_k"])
    source = load_json(
        Path(config["source_output_dir"]) / "metrics.json"
    )["evaluation"]["metrics"]
    for key, value in b0.items():
        if abs(value - source[key]) > 1e-6 or abs(b4[key] - value) > 1e-6:
            raise RuntimeError(f"B0/B4 không tái tạo CLS tại {key}")

    b0_correct = labels[indices[:, 0]].eq(labels)
    b3_correct = labels[b3_indices[:, 0]].eq(labels)
    improved = torch.nonzero(~b0_correct & b3_correct).flatten()
    regressed = torch.nonzero(b0_correct & ~b3_correct).flatten()

    def cases(query_indices):
        rows = []
        for query in query_indices[:200].tolist():
            rows.append(
                {
                    "query_index": query,
                    "query_path": paths_list[query],
                    "baseline_top1_index": int(indices[query, 0]),
                    "final_top1_index": int(b3_indices[query, 0]),
                }
            )
        return rows

    atomic_torch_save(
        {
            "candidate_indices": indices,
            "cosine_scores": cosine,
            "pair_confidence": confidence,
            "final_top_indices": b3_indices,
            "final_prefix_scores": b3_scores,
            "candidate_top_n": candidate_top_n,
            "lambda": weight,
        },
        paths["rankings"],
    )
    atomic_json_save(
        {
            "baseline_wrong_final_correct": cases(improved),
            "baseline_correct_final_wrong": cases(regressed),
            "counts": {
                "improved": len(improved),
                "regressed": len(regressed),
            },
        },
        paths["failures"],
    )
    metrics = load_json(paths["metrics"])
    metrics.update(
        {
            "status": "evaluated",
            "evaluation": {
                "b0_cls_cosine": b0,
                "b1_image_uncertainty": e1_metrics["evaluation"][
                    "e1_b_uncertainty_reranking"
                ],
                "b2_pair_confidence": b2,
                "b3_cosine_pair_confidence": b3,
                "b4_cosine_control": b4,
                "selected_candidate_top_n": candidate_top_n,
                "selected_lambda": weight,
                "num_queries": len(labels),
                "self_exclusion": True,
                "tie_policy": "stable_preserve_cosine_order",
                "device": str(device),
                "seconds": time.perf_counter() - started,
            },
        }
    )
    atomic_json_save(metrics, paths["metrics"])
    manifest["status"] = "evaluated"
    for path in (paths["rankings"], paths["failures"], paths["metrics"]):
        manifest["artifact_sha256"][path.name] = sha256_file(path)
    atomic_json_save(manifest, paths["manifest"])
    return metrics["evaluation"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate E2B.")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "cub_e2b.yaml")
    )
    args = parser.parse_args()
    print(run_e2b_evaluation(args.config))


if __name__ == "__main__":
    main()
