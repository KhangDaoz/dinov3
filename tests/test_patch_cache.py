import json

import pytest
import torch

from src.patch_cache import PatchShardDataset, validate_patch_manifest
from src.utils import sha256_file


def build_cache(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    shards = []
    for index, (start, stop) in enumerate(((0, 2), (2, 4))):
        path = split_dir / f"part-{index:05d}.pt"
        patches = torch.arange(start * 6, stop * 6, dtype=torch.float32).reshape(
            stop - start, 3, 2
        )
        torch.save({"patches": patches}, path)
        shards.append(
            {
                "path": str(path.relative_to(tmp_path)),
                "start": start,
                "stop": stop,
                "shape": list(patches.shape),
                "dtype": "float32",
                "sha256": sha256_file(path),
            }
        )
    section = {
        "count": 4,
        "labels": [0, 0, 1, 1],
        "paths": ["a", "b", "c", "d"],
        "shards": shards,
    }
    return {"format_version": 1, "splits": {"train": section, "test": section}}


def test_shard_dataset_crosses_boundaries_and_bounds_lru(tmp_path):
    manifest = build_cache(tmp_path)
    validate_patch_manifest(manifest, tmp_path)
    dataset = PatchShardDataset(manifest, "train", tmp_path, max_cached_shards=1)
    first, first_label = dataset[0]
    last, last_label = dataset[3]
    assert first.shape == (3, 2)
    assert first_label.item() == 0
    assert last_label.item() == 1
    assert len(dataset._cache) == 1
    assert not torch.equal(first, last)


def test_manifest_rejects_gap_or_overlap(tmp_path):
    manifest = build_cache(tmp_path)
    manifest["splits"]["train"]["shards"][1]["start"] = 1
    with pytest.raises(ValueError, match="ranges"):
        validate_patch_manifest(manifest, tmp_path, verify_hashes=False)


def test_manifest_detects_corrupt_shard(tmp_path):
    manifest = build_cache(tmp_path)
    path = tmp_path / manifest["splits"]["train"]["shards"][0]["path"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_patch_manifest(manifest, tmp_path)


def test_dataset_rejects_invalid_cache_limit(tmp_path):
    manifest = build_cache(tmp_path)
    with pytest.raises(ValueError, match="dương"):
        PatchShardDataset(manifest, "train", tmp_path, max_cached_shards=0)
