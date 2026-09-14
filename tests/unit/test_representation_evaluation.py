import pytest
import torch

from uncertainty_retrieval.evaluation.representation import (
    evaluate_top100,
    hits_from_ranking,
    select_representation_winner,
)


def test_evaluation_saves_candidate_ids_scores_and_excludes_self() -> None:
    features = torch.eye(5).repeat_interleave(2, dim=0)
    image_ids = torch.arange(100, 110)
    labels = torch.arange(10) // 2
    metrics, artifact = evaluate_top100(
        features, image_ids, labels, torch.device("cpu"), 3, ranking_depth=9
    )
    assert artifact["candidate_image_ids"].shape == (10, 9)
    assert artifact["cosine_scores"].shape == (10, 9)
    assert not torch.any(artifact["candidate_image_ids"] == image_ids[:, None])
    assert metrics["recall_at_1"] == pytest.approx(1.0)


def test_winner_rule_prioritizes_integer_hits_at_1() -> None:
    hits = {
        method: {f"hits_at_{k}": 50 for k in (1, 2, 4, 8)}
        for method in ("m1", "m2", "m3", "m4")
    }
    hits["m3"]["hits_at_1"] = 60
    hits["m1"]["hits_at_8"] = 100
    winner, _ = select_representation_winner(
        hits, {"m1": 0, "m2": 0, "m3": 10, "m4": 10}
    )
    assert winner == "m3"


def test_hits_are_recomputed_from_saved_ranking() -> None:
    features = torch.eye(5).repeat_interleave(2, dim=0)
    image_ids = torch.arange(100, 110)
    labels = torch.arange(10) // 2
    _, artifact = evaluate_top100(
        features, image_ids, labels, torch.device("cpu"), 3, ranking_depth=9
    )
    assert hits_from_ranking(artifact) == {
        "hits_at_1": 10, "hits_at_2": 10, "hits_at_4": 10, "hits_at_8": 10
    }
