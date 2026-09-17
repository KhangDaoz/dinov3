"""Distributed E2B evaluation and portable evidence export."""

import shutil
import sys
from pathlib import Path

import torch

from uncertainty_retrieval.config_e2b import load_e2b_config
from uncertainty_retrieval.data.e2b_controls import load_e1_control
from uncertainty_retrieval.data.feature_cache import sha256_file
from uncertainty_retrieval.data.pair_cache import load_pair_inputs, read_json, verify_provenance
from uncertainty_retrieval.evaluation.pair_confidence import (
    constrained_fusion_order, paired_bootstrap_chunked,
    ranking_metrics, raw_alpha_rankings, reliability, select_lambda,
    write_grid_svg, write_reliability_svg,
)
from uncertainty_retrieval.evaluation.retrieval import cosine_rankings
from uncertainty_retrieval.evaluation.visualization import write_failure_contact_sheet
from uncertainty_retrieval.models.pair_confidence import PairConfidenceNetwork
from uncertainty_retrieval.training.pair_confidence import setup, merge_shards, score_pairs
from uncertainty_retrieval.training.representation import atomic_torch_save
from uncertainty_retrieval.utils import cleanup_distributed, distributed_barrier, environment_metadata, write_json


def export_e2b(root: Path, split: str) -> None:
    required = ["selection.json", "checkpoints/best.pt", "config_resolved.yaml",
                f"scores/{split}_top100.pt", f"metrics/{split}.json",
                f"metrics/{split}_reliability.json",
                f"metrics/{split}_topn_lambda_grid.json",
                f"metrics/{split}_method_topn.json",
                f"failure_cases/{split}.json",
                f"figures/{split}_topn_lambda_recall1.svg"]
    required += [f"rankings/{split}_{method}.pt" for method in ("baseline", "pair", "fusion")]
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete E2B export:\n" + "\n".join(missing))
    destination = root / "export"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("scores", "rankings", "metrics", "failure_cases", "controls",
                 "inputs", "split", "checkpoints", "figures"):
        if (root / name).is_dir():
            shutil.copytree(root / name, destination / name, dirs_exist_ok=True)
    for name in ("selection.json", "config_resolved.yaml", "metadata.json",
                 "environment.json", "report.tex"):
        if (root / name).is_file():
            shutil.copy2(root / name, destination / name)
    write_json({"stage":split,"checkpoint_sha256":sha256_file(root/"checkpoints/best.pt"),
                "required_artifact_hashes":{name:sha256_file(root/name) for name in required}},
               destination/f"manifest_{split}.json")
    print(f"Complete E2B evidence bundle: {destination.resolve()}", flush=True)


def _fusion_grid(candidates, cosine, confidence, labels, top_ns, lambdas):
    metrics, orders = {}, {}
    for top_n in top_ns:
        row, row_orders = {}, {}
        for coefficient in lambdas:
            order, _ = constrained_fusion_order(
                cosine, confidence, coefficient, top_n
            )
            key = str(coefficient)
            row[key] = ranking_metrics(candidates.gather(1, order), labels)
            row_orders[key] = order
        metrics[str(top_n)] = row
        orders[str(top_n)] = row_orders
    return metrics, orders


def _failure_cases(view, ids, labels, candidates, reranked, cosine,
                   confidence, ranked_scores, rerank_order, limit=20):
    baseline_hit = labels[candidates[:, 0]].eq(labels)
    reranked_hit = labels[reranked[:, 0]].eq(labels)
    groups = {key: [] for key in (
        "wrong_to_correct", "correct_to_wrong", "still_wrong", "still_correct"
    )}
    for row in ids.argsort().tolist():
        if baseline_hit[row]:
            category = "still_correct" if reranked_hit[row] else "correct_to_wrong"
        else:
            category = "wrong_to_correct" if reranked_hit[row] else "still_wrong"
        if len(groups[category]) >= limit:
            continue
        top = reranked[row, :8]
        top_positions = rerank_order[row, :8]
        baseline = candidates[row, :8]
        groups[category].append({
            "category": category,
            "query_id": int(ids[row]),
            "query_label": int(labels[row]),
            "query_path": view["paths"][row],
            "baseline_ids": ids[baseline].tolist(),
            "baseline_labels": labels[baseline].tolist(),
            "baseline_paths": [view["paths"][i] for i in baseline.tolist()],
            "reranked_ids": ids[top].tolist(),
            "reranked_labels": labels[top].tolist(),
            "reranked_paths": [view["paths"][i] for i in top.tolist()],
            "baseline_cosine": cosine[row, :8].tolist(),
            "reranked_cosine": cosine[row].gather(0, top_positions).tolist(),
            "reranked_confidence": confidence[row].gather(0, top_positions).tolist(),
            "reranked_scores": ranked_scores[row].gather(0, top_positions).tolist(),
        })
    return groups


def evaluate(config_path: Path, split: str):
    if split not in {"validation", "test"}:
        raise ValueError("E2B evaluation supports validation/test only")
    config = load_e2b_config(config_path)
    rank, world, device = setup(config)
    root = Path(config.output_root)
    try:
        view, provenance = load_pair_inputs(config, config_path, split)
        for name in ("fit", "validation"):
            provenance[f"{name}_mining_sha256"] = sha256_file(root / "pairs" / f"{name}_static.pt")
        checkpoint_path = root / "checkpoints/best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        verify_provenance(checkpoint["provenance"], provenance)
        checkpoint_hash = sha256_file(checkpoint_path)
        selection = None
        if split == "test":
            selection = read_json(root / "selection.json")
            verify_provenance(selection["provenance"], provenance)
            if selection["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Checkpoint changed after validation selection")
            if selection["lambda"] not in config.evaluation.lambdas or selection["N"] != 100:
                raise ValueError("Invalid frozen E2B tuning record")
            if sha256_file(root / "metrics/validation_grid.json") != selection["grid_sha256"]:
                raise ValueError("Validation grid changed after tuning")
            for relative_path, expected in selection.get("control_hashes", {}).items():
                if sha256_file(root / relative_path) != expected:
                    raise ValueError(f"E1 control changed after tuning: {relative_path}")
        features = view["features"].to(device)
        ids = view["image_ids"]
        positions = {int(value): i for i, value in enumerate(ids)}
        if split == "validation":
            pool = torch.load(root / "pairs/validation_static.pt", map_location="cpu", weights_only=True)
            if not torch.equal(pool["image_ids"], ids):
                raise ValueError("Validation mining IDs mismatch")
            candidates, cosine = pool["candidates"], pool["cosine"]
        else:
            accepted = view["accepted_ranking"]
            candidates = torch.tensor([[positions[int(value)] for value in row]
                                       for row in accepted["candidate_image_ids"]])
            cosine = accepted["cosine_scores"]
        rows = torch.arange(rank, len(ids), world)
        model = PairConfidenceNetwork().to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        logits = score_pairs(model, features, rows, candidates[rows], config.runtime.pair_chunk_size)
        # Orientation diagnostic only; never used in confidence/fusion.
        reversed_logits = []
        local_candidates = candidates[rows].flatten().to(device)
        local_queries = rows.repeat_interleave(100).to(device)
        with torch.inference_mode():
            for start in range(0, len(local_queries), config.runtime.pair_chunk_size):
                end = start + config.runtime.pair_chunk_size
                reversed_logits.append(model(features[local_candidates[start:end]],
                                             features[local_queries[start:end]]).cpu())
        shard = {"rows": rows, "logits": logits.cpu(),
                 "swapped_logits": torch.cat(reversed_logits).reshape(len(rows), 100)}
        fields = ["logits", "swapped_logits"]
        uncertainty = control_provenance = None
        if config.controls.enabled:
            alpha, uncertainty, control_provenance = load_e1_control(config, split, view, provenance, device)
            alpha = alpha.to(device)
            a1_indices, a1_distances = raw_alpha_rankings(alpha, ids.to(device), rows.to(device),
                                                        config.runtime.query_chunk_size)
            a2 = cosine_rankings(alpha[rows.to(device)], alpha, ids[rows].to(device), ids.to(device),
                                 config.runtime.query_chunk_size, 100)
            shard.update(a1_indices=a1_indices, a1_distances=a1_distances,
                         a2_indices=a2.indices, a2_cosine=a2.scores)
            fields.extend(("a1_indices", "a1_distances", "a2_indices", "a2_cosine"))
        shards = root / "scores/shards"
        atomic_torch_save(shard, shards / f"{split}_rank{rank}.pt")
        distributed_barrier(device)
        # Distributed scoring is finished. Tear down on ALL ranks before
        # rank-0-only sorting/report/export: peers must not sit in an NCCL
        # barrier if postprocessing raises or takes longer than its timeout.
        cleanup_distributed()
        if rank == 0:
            merged = merge_shards(shards, split, world, fields)
            confidence = merged["logits"].sigmoid()
            labels = view["labels"]
            baseline_metrics = ranking_metrics(candidates, labels)
            if split == "test":
                accepted_hits = read_json(config.inputs.selection)["test_hits"]["m1"]
                if any(baseline_metrics[key] != value for key, value in accepted_hits.items()):
                    raise ValueError("E2B B0 does not reproduce accepted M1")
            full_grid, grid_orders = _fusion_grid(
                candidates, cosine, confidence, labels,
                config.evaluation.uncertainty_top_n,
                config.evaluation.lambdas,
            )
            grid = full_grid[str(config.evaluation.ranking_depth)]
            write_json(
                full_grid,
                root / "metrics" / f"{split}_topn_lambda_grid.json",
            )
            write_grid_svg(
                full_grid,
                root / "figures" / f"{split}_topn_lambda_recall1.svg",
                f"E2B {split}: Recall@1 by Top-N and lambda",
            )
            uncertainty_n = None
            if split == "validation":
                coefficient = select_lambda(grid)
                write_json(grid, root / "metrics/validation_grid.json")
                if uncertainty is not None:
                    u_grid = {}
                    for n in config.evaluation.uncertainty_top_n:
                        u_order = uncertainty[candidates[:, :n]].argsort(dim=1, stable=True)
                        reranked = candidates.clone()
                        reranked[:, :n] = candidates[:, :n].gather(1, u_order)
                        u_grid[str(n)] = ranking_metrics(reranked, labels)
                    uncertainty_n = int(max(u_grid, key=lambda n: (
                        *(u_grid[n][f"hits_at_{k}"] for k in (1,2,4,8)), -int(n))))
                    write_json(u_grid, root / "controls/validation_u1_grid.json")
                selection = {"schema_version": 1, "lambda": coefficient, "N": 100,
                             "epoch": checkpoint["epoch"], "checkpoint_sha256": checkpoint_hash,
                             "provenance": provenance, "u1_N": uncertainty_n,
                             "grid_sha256": sha256_file(root / "metrics/validation_grid.json"),
                             "validation_grid": grid,
                             "decision_trace": sorted(
                                 [{"lambda":float(value),"hits":[grid[value][f"hits_at_{k}"]
                                                              for k in (1,2,4,8)]}
                                  for value in grid],
                                 key=lambda item:(*item["hits"],item["lambda"]),reverse=True),
                             "tie_rule": "Hits@1 -> Hits@2 -> Hits@4 -> Hits@8 -> larger lambda",
                             "notice": "Benchmark already used for E2A representation selection."}
                if config.controls.enabled and config.controls.source == "controlled_retrain":
                    selection["control_hashes"] = {
                        path: sha256_file(root / path) for path in (
                            "controls/edl/tuning/uncertainty/validation.pt",
                            "controls/edl/tuning/best.pt", "controls/validation_u1_grid.json")
                    }
                write_json(selection, root / "selection.json")
            coefficient = selection["lambda"]
            selected_n = int(selection["N"])
            order = grid_orders[str(selected_n)][str(coefficient)]
            pair_order = grid_orders[str(selected_n)][str(0.0)]
            fused_scores = coefficient * cosine + (1 - coefficient) * confidence
            rankings = {"baseline": candidates, "pair": candidates.gather(1, pair_order),
                        "fusion": candidates.gather(1, order)}
            metric_table = {name: ranking_metrics(value, labels) for name, value in rankings.items()}
            method_topn = {
                "baseline": baseline_metrics,
                "pair_wise_confidence": {
                    str(n): full_grid[str(n)][str(0.0)]
                    for n in config.evaluation.uncertainty_top_n
                },
                "fusion": {
                    str(n): {
                        str(value): full_grid[str(n)][str(value)]
                        for value in config.evaluation.lambdas
                    }
                    for n in config.evaluation.uncertainty_top_n
                },
            }
            metric_table["candidate_positive_coverage"] = float(
                labels[candidates].eq(labels[:, None]).any(1).float().mean())
            for name, indices in rankings.items():
                ranked_scores = cosine if name == "baseline" else (
                    confidence.gather(1, pair_order) if name == "pair" else fused_scores.gather(1, order))
                atomic_torch_save({"query_image_ids": ids, "query_labels": labels,
                                   "candidate_image_ids": ids[indices], "scores": ranked_scores,
                                   "score_type": name, "split": split, "ranking_depth": 100,
                                   "self_match_exclusion": True,
                                   "hits": {f"hits_at_{k}": labels[indices[:, :k]].eq(labels[:, None]).any(1)
                                            for k in (1,2,4,8)}}, root / "rankings" / f"{split}_{name}.pt")
            for n in config.evaluation.uncertainty_top_n:
                for value in config.evaluation.lambdas:
                    grid_order = grid_orders[str(n)][str(value)]
                    grid_ranking = candidates.gather(1, grid_order)
                    slug = f"{value:g}".replace(".", "p")
                    atomic_torch_save(
                        {
                            "schema_version": 1,
                            "query_image_ids": ids,
                            "query_labels": labels,
                            "candidate_image_ids": ids[grid_ranking],
                            "split": split,
                            "N": n,
                            "lambda": value,
                            "metrics": full_grid[str(n)][str(value)],
                            "selection_role": "analysis_grid",
                        },
                        root / "rankings" / f"{split}_fusion_n{n}_lambda{slug}.pt",
                    )
            targets = labels[candidates].eq(labels[:, None]).float()
            atomic_torch_save({"schema_version": 1, "query_image_ids": ids,
                               "candidate_image_ids": ids[candidates], "cosine": cosine,
                               "logits": merged["logits"], "pair_confidence": confidence,
                               "targets": targets, "fusion_scores": fused_scores,
                               "lambda": coefficient, "N": selected_n, "split": split,
                               "checkpoint_sha256": checkpoint_hash, "provenance": provenance},
                              root / "scores" / f"{split}_top100.pt")
            reliability_table = {
                "candidate_pairs": reliability(confidence.flatten(), targets.flatten(), merged["logits"].flatten()),
                "baseline_top1_pair": reliability(confidence[:,0], targets[:,0], merged["logits"][:,0]),
                "baseline_top1_cosine_mapped_to_01": reliability((cosine[:,0]+1)/2, targets[:,0]),
                "post_fusion_top1_pair": reliability(confidence.gather(1,order)[:,0], targets.gather(1,order)[:,0]),
                "swap_mean_absolute_difference": float((confidence-merged["swapped_logits"].sigmoid()).abs().mean()),
                "notice": "Balanced mined training; sigmoid is not automatically calibrated."}
            u_rankings = {}
            if uncertainty is not None:
                n = selection["u1_N"]
                if n not in config.evaluation.uncertainty_top_n:
                    raise ValueError("Missing validation-selected U1 budget")
                u_grid, u_rankings = {}, {}
                for budget in config.evaluation.uncertainty_top_n:
                    u_order = uncertainty[candidates[:, :budget]].argsort(
                        dim=1, stable=True
                    )
                    reranked = candidates.clone()
                    reranked[:, :budget] = candidates[:, :budget].gather(1, u_order)
                    u_grid[str(budget)] = ranking_metrics(reranked, labels)
                    u_rankings[str(budget)] = reranked
                    atomic_torch_save(
                        {
                            "schema_version": 1,
                            "query_image_ids": ids,
                            "query_labels": labels,
                            "candidate_image_ids": ids[reranked],
                            "candidate_uncertainty": uncertainty[reranked],
                            "split": split,
                            "N": budget,
                            "metrics": u_grid[str(budget)],
                        },
                        root / "controls" / f"{split}_U1_n{budget}.pt",
                    )
                write_json(u_grid, root / "controls" / f"{split}_u1_grid.json")
                method_topn["image_uncertainty"] = u_grid
                # The official cross-method comparison uses the common N=100.
                # u1_N remains the validation-best diagnostic budget only.
                u_indices = u_rankings[str(selected_n)]
                metric_table.update(U1=ranking_metrics(u_indices,labels),
                                    A1=ranking_metrics(merged["a1_indices"],labels),
                                    A2=ranking_metrics(merged["a2_indices"],labels))
                reliability_table["baseline_top1_image_certainty"] = reliability(1-uncertainty[candidates[:,0]], targets[:,0])
                for name, indices, values, field in (
                    ("A1", merged["a1_indices"], merged["a1_distances"], "euclidean_distances"),
                    ("A2", merged["a2_indices"], merged["a2_cosine"], "cosine_scores"),
                    ("U1", u_indices, uncertainty[u_indices], "candidate_uncertainty")):
                    atomic_torch_save({"query_image_ids":ids,"query_labels":labels,
                                       "candidate_image_ids":ids[indices],field:values,
                                       "split":split,"provenance":control_provenance},
                                      root / "controls" / f"{split}_{name}.pt")
                write_json(control_provenance, root / "inputs" / f"e1_{split}_provenance.json")
            else:
                metric_table["controls_unavailable"] = "Explicitly disabled in config; no historical score substitution."
            write_json(method_topn, root / "metrics" / f"{split}_method_topn.json")
            write_json(metric_table, root / "metrics" / f"{split}.json")
            write_json(reliability_table, root / "metrics" / f"{split}_reliability.json")
            write_reliability_svg(reliability_table["candidate_pairs"],
                                  root / "figures" / f"{split}_candidate_calibration.svg")
            write_reliability_svg(reliability_table["baseline_top1_pair"],
                                  root / "figures" / f"{split}_top1_calibration.svg")
            failures = _failure_cases(
                view, ids, labels, candidates, rankings["fusion"], cosine,
                confidence, fused_scores, order,
            )
            write_json(failures, root / "failure_cases" / f"{split}.json")
            for category, cases in failures.items():
                write_failure_contact_sheet(
                    cases,
                    config.dataset.root,
                    root / "figures" / "failure_cases" / f"{split}_{category}.png",
                    f"{split}: {category} (N={selected_n}, lambda={coefficient:g})",
                )
            # Method- and budget-specific qualitative evidence. Selection is
            # never based on these images; rows are deterministic by image ID.
            for budget in config.evaluation.uncertainty_top_n:
                method_orders = {
                    "pair": grid_orders[str(budget)][str(0.0)],
                    "fusion": grid_orders[str(budget)][str(coefficient)],
                }
                if u_rankings:
                    method_orders["image_uncertainty"] = None
                for method, method_order in method_orders.items():
                    if method == "image_uncertainty":
                        method_ranking = u_rankings[str(budget)]
                        # Recover positions within the original cosine list.
                        lookup = [{int(value): position for position, value in enumerate(row)}
                                  for row in candidates.tolist()]
                        method_order = torch.tensor([
                            [lookup[row][int(value)] for value in ranked]
                            for row, ranked in enumerate(method_ranking.tolist())
                        ])
                        method_scores = -uncertainty[candidates]
                    else:
                        method_ranking = candidates.gather(1, method_order)
                        value = 0.0 if method == "pair" else coefficient
                        method_scores = value * cosine + (1 - value) * confidence
                    method_failures = _failure_cases(
                        view, ids, labels, candidates, method_ranking, cosine,
                        confidence, method_scores, method_order,
                    )
                    write_json(
                        method_failures,
                        root / "failure_cases" / f"{split}_{method}_n{budget}.json",
                    )
                    for category, cases in method_failures.items():
                        write_failure_contact_sheet(
                            cases,
                            config.dataset.root,
                            root / "figures" / "failure_cases"
                            / f"{split}_{method}_n{budget}_{category}.png",
                            f"{split}: {method}, N={budget}, {category}",
                        )
            if split == "test":
                bootstrap = {}
                for k in (1,2,4,8):
                    b = labels[candidates[:,:k]].eq(labels[:,None]).any(1)
                    p = labels[rankings["fusion"][:,:k]].eq(labels[:,None]).any(1)
                    bootstrap[str(k)] = paired_bootstrap_chunked(b,p,device,config.evaluation.bootstrap_samples)
                write_json(bootstrap, root / "metrics/test_bootstrap.json")
                report_methods = (["baseline", "A1", "U1"] if uncertainty is not None else ["baseline"])
                report_methods += ["pair", "fusion"]
                rows_tex = "\n".join(name + " & " + " & ".join(
                    f"{100*metric_table[name][f'recall_at_{k}']:.2f}" for k in (1,2,4,8)
                ) + r" \\" for name in report_methods)
                report = (r"\documentclass{article}\usepackage{booktabs}\begin{document}" +
                          "\nE2B single-seed (42); inherited E2A test-selection bias.\n" +
                          f"Selected epoch: {checkpoint['epoch']}; lambda: {coefficient}.\n" +
                          r"\begin{tabular}{lrrrr}\toprule Method & R@1 & R@2 & R@4 & R@8 \\ \midrule" +
                          "\n" + rows_tex + "\n" + r"\bottomrule\end{tabular}\end{document}")
                (root / "report.tex").write_text(report,encoding="utf-8")
            write_json(environment_metadata(sys.argv), root / "metrics" / f"{split}_environment.json")
            export_e2b(root, split)
            print(f"{split}: lambda={coefficient}; fusion={metric_table['fusion']}",flush=True)
    finally:
        cleanup_distributed()
