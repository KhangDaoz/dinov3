from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    from .utils import sha256_file
except ImportError:
    from utils import sha256_file


def validate_patch_manifest(manifest, cache_root, verify_hashes=True):
    if manifest.get("format_version") != 1:
        raise ValueError("Patch manifest format_version không được hỗ trợ")
    cache_root = Path(cache_root)
    for split in ("train", "test"):
        section = manifest.get("splits", {}).get(split)
        if not isinstance(section, dict):
            raise ValueError(f"Manifest thiếu split {split}")
        expected_start = 0
        for shard in section.get("shards", []):
            if shard["start"] != expected_start or shard["stop"] <= shard["start"]:
                raise ValueError(f"Shard ranges {split} bị thiếu, lặp hoặc chồng lấn")
            path = cache_root / shard["path"]
            if not path.is_file():
                raise FileNotFoundError(f"Thiếu patch shard: {path}")
            if verify_hashes and sha256_file(path) != shard["sha256"]:
                raise ValueError(f"Patch shard checksum không khớp: {path}")
            expected_start = shard["stop"]
        if expected_start != section.get("count"):
            raise ValueError(f"Shard count {split} không khớp manifest")
        if len(section.get("labels", [])) != expected_start:
            raise ValueError(f"Labels {split} không khớp shard rows")
        if len(section.get("paths", [])) != expected_start:
            raise ValueError(f"Paths {split} không khớp shard rows")


class PatchShardDataset(Dataset):
    """Lazy fixed-shard patch dataset with a bounded per-process LRU."""

    def __init__(self, manifest, split, cache_root, max_cached_shards=4):
        if split not in ("train", "test"):
            raise ValueError("Patch split phải là train hoặc test")
        if max_cached_shards <= 0:
            raise ValueError("max_cached_shards phải dương")
        self.section = manifest["splits"][split]
        self.cache_root = Path(cache_root)
        self.max_cached_shards = max_cached_shards
        self.shards = self.section["shards"]
        self.stops = [shard["stop"] for shard in self.shards]
        self.labels = torch.tensor(self.section["labels"], dtype=torch.long)
        self.paths = list(self.section["paths"])
        self._cache = OrderedDict()

    def __len__(self):
        return self.section["count"]

    def _load_shard(self, shard_index):
        if shard_index in self._cache:
            payload = self._cache.pop(shard_index)
            self._cache[shard_index] = payload
            return payload
        payload = torch.load(
            self.cache_root / self.shards[shard_index]["path"],
            map_location="cpu",
            weights_only=True,
        )
        patches = payload.get("patches")
        expected_shape = tuple(self.shards[shard_index]["shape"])
        if not isinstance(patches, torch.Tensor) or tuple(patches.shape) != expected_shape:
            raise ValueError(f"Patch shard {shard_index} có shape không hợp lệ")
        self._cache[shard_index] = patches
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return patches

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.stops, index)
        shard = self.shards[shard_index]
        patches = self._load_shard(shard_index)
        return patches[index - shard["start"]], self.labels[index]
