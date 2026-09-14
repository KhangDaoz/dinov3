import json

import pytest
import torch

from uncertainty_retrieval.evaluation.representation import (
    evaluate_top100,
    select_representation_winner,
    verify_selection_lock,
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


def test_winner_rule_prioritizes_recall_at_1() -> None:
    metrics = {
        method: {f"recall_at_{k}": 0.5 for k in (1, 2, 4, 8)}
        for method in ("m1", "m2", "m3", "m4")
    }
    metrics["m3"]["recall_at_1"] = 0.6
    metrics["m1"]["recall_at_8"] = 1.0
    winner, _ = select_representation_winner(
        metrics, {"m1": 0, "m2": 0, "m3": 10, "m4": 10}
    )
    assert winner == "m3"


def test_final_test_fails_closed_without_complete_lock(tmp_path) -> None:
    with pytest.raises(PermissionError, match="locked"):
        verify_selection_lock(tmp_path / "missing.json")
    path = tmp_path / "selection_lock.json"
    path.write_text(json.dumps({"test_unlocked": True}), encoding="utf-8")
    with pytest.raises(PermissionError, match="incomplete"):
        verify_selection_lock(path)
