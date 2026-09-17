"""Candidate-constrained fusion, raw-alpha L2 and reliability metrics."""

import torch
import torch.nn.functional as F

from .evidential import binary_auroc, binary_auprc, aurc


def fusion_order(cosine, confidence, coefficient: float):
    if cosine.shape != confidence.shape or cosine.ndim != 2:
        raise ValueError("Pair scores must be matching matrices")
    if not 0 <= coefficient <= 1 or not torch.isfinite(confidence).all():
        raise ValueError("Invalid fusion inputs")
    if ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Confidence must be in [0,1]")
    score = coefficient * cosine + (1 - coefficient) * confidence
    if coefficient == 1:
        order = torch.arange(cosine.shape[1], device=cosine.device).expand_as(cosine)
    else:
        order = torch.argsort(score, descending=True, stable=True, dim=1)
    return order, score


def constrained_fusion_order(cosine, confidence, coefficient: float, top_n: int):
    """Rank only the first ``top_n`` cosine candidates and preserve the tail."""
    if not 0 < top_n <= cosine.shape[1]:
        raise ValueError("top_n exceeds candidate depth")
    local_order, score = fusion_order(
        cosine[:, :top_n], confidence[:, :top_n], coefficient
    )
    tail = torch.arange(
        top_n, cosine.shape[1], device=cosine.device
    ).expand(cosine.shape[0], -1)
    return torch.cat((local_order, tail), dim=1), score


def ranking_metrics(candidates, labels):
    relevant = labels[candidates].eq(labels[:, None])
    hits = {f"hits_at_{k}": int(relevant[:, :k].any(1).sum()) for k in (1, 2, 4, 8)}
    return {**hits, **{f"recall_at_{k}": hits[f"hits_at_{k}"] / len(labels)
                       for k in (1, 2, 4, 8)}}


def select_lambda(grid: dict) -> float:
    return float(max(grid, key=lambda value: (
        *(grid[value][f"hits_at_{k}"] for k in (1, 2, 4, 8)), float(value)
    )))


def raw_alpha_rankings(alpha, image_ids, query_positions, chunk_size=128, depth=100):
    """Full-gallery Euclidean ranking, without alpha normalization."""
    if alpha.ndim != 2 or not torch.isfinite(alpha).all() or (alpha < 1).any():
        raise ValueError("Alpha must be finite [N,K] with entries >=1")
    if not 0 < depth < len(alpha):
        raise ValueError("Invalid alpha ranking depth")
    indices, distances = [], []
    for start in range(0, len(query_positions), chunk_size):
        positions = query_positions[start:start + chunk_size]
        # Direct distances avoid cancellation for large, nearly equal alpha.
        distance = torch.cdist(alpha[positions].float(), alpha.float(),
                               compute_mode="donot_use_mm_for_euclid_dist")
        self_mask = image_ids[positions, None].eq(image_ids[None, :])
        distance.masked_fill_(self_mask, torch.inf)
        order = distance.argsort(dim=1, stable=True)[:, :depth]
        indices.append(order.cpu())
        distances.append(distance.gather(1, order).cpu())
    return torch.cat(indices), torch.cat(distances)


def reliability(scores, targets, logits=None):
    scores, targets = scores.detach().cpu().float(), targets.detach().cpu().float()
    if scores.shape != targets.shape or scores.ndim != 1 or not len(scores):
        raise ValueError("Invalid reliability population")
    result = {"count": len(scores), "positive_prevalence": float(targets.mean())}
    for name, function in (("auroc", binary_auroc), ("auprc", binary_auprc)):
        try:
            result[name] = function(scores, targets.long())
        except ValueError as error:
            result[name] = {"value": None, "reason": str(error)}
    result["brier"] = float((scores - targets).square().mean())
    result["bce"] = float(F.binary_cross_entropy_with_logits(
        logits.detach().cpu().float(), targets
    ) if logits is not None else F.binary_cross_entropy(scores.clamp(1e-7, 1 - 1e-7), targets))
    bins, ece = [], 0.0
    indices = (scores * 10).long().clamp(0, 9)
    for index in range(10):
        mask = indices == index
        count = int(mask.sum())
        predicted = float(scores[mask].mean()) if count else None
        observed = float(targets[mask].mean()) if count else None
        if count:
            ece += count / len(scores) * abs(predicted - observed)
        bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                     "count": count, "predicted": predicted, "observed": observed})
    result.update(ece_10_bins=ece, reliability_bins=bins,
                  aurc=aurc(1 - scores, (1 - targets).long()))
    order = scores.argsort(descending=True,stable=True)
    retained = torch.arange(1,len(scores)+1,dtype=torch.float32)
    risks = (1-targets[order]).cumsum(0)/retained
    points = torch.linspace(0,len(scores)-1,min(100,len(scores))).long()
    result["risk_coverage"] = {"coverage":(retained[points]/len(scores)).tolist(),
                               "risk":risks[points].tolist()}
    return result


def write_reliability_svg(metrics: dict, path) -> None:
    """Render fixed-bin calibration diagnostics without plotting dependencies."""
    points = []
    for item in metrics["reliability_bins"]:
        if item["count"]:
            x = 40 + 320*item["predicted"]
            y = 360 - 320*item["observed"]
            points.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="navy"/>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="420" height="420" '
           'viewBox="0 0 420 420"><rect width="420" height="420" fill="white"/>'
           '<path d="M40 40 V360 H360" fill="none" stroke="black"/>'
           '<path d="M40 360 L360 40" stroke="gray" stroke-dasharray="5 5"/>'
           '<text x="70" y="395">Predicted compatibility (0 to 1)</text>'
           '<text x="40" y="25">Observed positive fraction (0 to 1)</text>'
           + ''.join(points) + '</svg>')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(svg,encoding="utf-8")


def write_grid_svg(grid: dict, path, title: str) -> None:
    """Write an R@1 plot for every Top-N/lambda combination as portable SVG."""
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
    top_ns = sorted(int(value) for value in grid)
    lambdas = sorted({float(value) for row in grid.values() for value in row})
    values = [grid[str(n)][str(value)]["recall_at_1"] for n in top_ns for value in lambdas]
    lower = max(0.0, min(values) - 0.02)
    upper = min(1.0, max(values) + 0.02)
    span = max(upper - lower, 1e-6)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="500" viewBox="0 0 760 500">',
        '<rect width="760" height="500" fill="white"/>',
        f'<text x="380" y="28" text-anchor="middle" font-size="18">{title}</text>',
        '<path d="M70 55 V420 H690" fill="none" stroke="black"/>',
        '<text x="380" y="470" text-anchor="middle">Reranking depth N</text>',
        '<text x="18" y="245" transform="rotate(-90 18 245)" text-anchor="middle">Recall@1</text>',
    ]
    for tick in range(6):
        value = lower + span * tick / 5
        y = 420 - 365 * tick / 5
        parts.append(f'<path d="M65 {y:.1f} H690" stroke="#dddddd"/>')
        parts.append(f'<text x="58" y="{y + 4:.1f}" text-anchor="end" font-size="11">{100*value:.2f}%</text>')
    for index, coefficient in enumerate(lambdas):
        points = []
        for column, n in enumerate(top_ns):
            x = 90 + 580 * column / max(1, len(top_ns) - 1)
            value = grid[str(n)][str(coefficient)]["recall_at_1"]
            y = 420 - 365 * (value - lower) / span
            points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[index % len(colors)]}"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[index % len(colors)]}" stroke-width="2"/>')
        legend_y = 60 + 20 * index
        parts.append(f'<path d="M705 {legend_y} h18" stroke="{colors[index % len(colors)]}" stroke-width="3"/>')
        parts.append(f'<text x="728" y="{legend_y + 4}" font-size="11">lambda={coefficient:g}</text>')
    for column, n in enumerate(top_ns):
        x = 90 + 580 * column / max(1, len(top_ns) - 1)
        parts.append(f'<text x="{x:.1f}" y="440" text-anchor="middle">{n}</text>')
    parts.append('</svg>')
    path = __import__('pathlib').Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(parts), encoding='utf-8')


def paired_bootstrap_chunked(baseline, proposed, device, samples=2000, seed=42):
    delta = proposed.float().to(device) - baseline.float().to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    means = []
    for start in range(0, samples, 64):
        indices = torch.randint(len(delta), (min(64, samples - start), len(delta)),
                                generator=generator, device=device)
        means.append(delta[indices].mean(1))
    values = torch.cat(means)
    return {"delta": float(delta.mean()), "lower": float(values.quantile(0.025)),
            "upper": float(values.quantile(0.975)), "samples": samples,
            "notice": "Exploratory: inherited test selection bias; single seed."}
