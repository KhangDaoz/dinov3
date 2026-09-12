# E2A-M1 Implementation Plan — CLS Baseline

## Objective and Scope

Build a reproducible frozen DINOv3 ViT-B/16 baseline that extracts the final-layer CLS token, L2-normalizes it, performs cosine retrieval on the CUB-200-2011 evaluation split, and reports Recall@1/2/4/8. M1 must not train parameters, estimate uncertainty, rerank results, or add projection layers.

The Proxy Anchor CUB protocol is the retrieval predecessor. EDL, IDML, and Evidential Transformers remain deferred to uncertainty-aware phases; importing their uncertainty logic into M1 would invalidate the baseline.

## Preconditions

1. Use the repository's existing `.venv` with Python 3.14.7. The pinned PyTorch, torchvision, Transformers, and supporting packages have been import-verified in this environment.
2. Keep the Hugging Face token in `configs/cub_e2a.yaml` and load it through the existing `token` key. Redact it only from logs and generated metadata.
3. Confirm the pinned model revision is accessible and cache it before the full run.
4. Validate the class-disjoint split: classes 0–99 contain 5,864 train images and classes 100–199 contain 5,924 eval images.

## Implementation Work Packages

### WP1 — Configuration and shared utilities

- Add strict config loading, seed/thread setup, device resolution, output-directory creation, JSON serialization, timings, and SHA-256 calculation in `src/utils.py`.
- Use `device: auto`: select CUDA whenever available and fall back to CPU only when no GPU is available. Keep tensors on CPU only for persisted artifacts or operations that explicitly require CPU.
- Reject unsupported representations, splits, devices, and non-positive batch/chunk sizes with actionable messages.
- Store the fully resolved, secret-free config in metadata.

### WP2 — Dataset and batching

- Retain the class-based `train`/`eval` protocol in `src/dataset.py` and expose stable image paths and integer labels.
- Add a collate path that keeps PIL images as a list; apply the checkpoint's `AutoImageProcessor` once per batch.
- Assert class ranges, sample counts, RGB conversion, and path/label ordering.

### WP3 — Frozen CLS extraction

- Refactor `src/dinov3_backbone.py` into reusable loading and inference helpers while preserving its smoke-check CLI.
- Add `src/representations.py` with an explicit `cls` selector: `last_hidden_state[:, 0, :]` followed by L2 normalization.
- Use `torch.inference_mode()`, model evaluation mode, and no trainable backbone parameters. Accumulate CPU `float32` tensors in deterministic dataset order.

### WP4 — Artifacts and extraction CLI

- Implement `scripts/extract_features.py` to process both splits and atomically write:
  - `outputs/cls/train_embeddings.pt` with shape `[5864, 768]`;
  - `outputs/cls/test_embeddings.pt` with shape `[5924, 768]`;
  - `outputs/cls/labels.pt` containing labels and paths for both splits;
  - `outputs/cls/metrics.json` with `status: extracted`, runtime/device/dtype, processor settings, model revision, dimensions, counts, and cache hashes.
- Refuse partial/mismatched caches unless an explicit overwrite option is supplied.

### WP5 — Retrieval and evaluation

- Implement chunked cosine search in `src/retrieval.py`; normalized dot product is the similarity. Run retrieval on the resolved GPU when available.
- Use every eval image as both query and gallery, mask only the identical sample, and define a deterministic tie policy.
- Implement `scripts/evaluate.py` to validate cache integrity and append Recall@1/2/4/8, query/gallery counts, exclusion policy, and timing to `metrics.json` with `status: evaluated`.
- Implement `scripts/run_e2a.py` as the extract-then-evaluate orchestrator without duplicating core logic.

### WP6 — Tests and documentation

- Add unit tests for split membership, invalid modes, CLS slicing, normalization, cache alignment, self-exclusion, Recall@K, chunk equivalence, and corrupt metadata.
- Use synthetic tensors and mocked model outputs; keep unit tests independent of CUB and network access.
- Document exact commands, artifact schema, and a result table in the README after a verified run.

## Verification Sequence

```bash
.venv/bin/python -m compileall src scripts
.venv/bin/python -m pytest -q
.venv/bin/python src/dinov3_backbone.py --config configs/cub_e2a.yaml --root data
.venv/bin/python scripts/extract_features.py --config configs/cub_e2a.yaml
.venv/bin/python scripts/evaluate.py --config configs/cub_e2a.yaml
```

Run a small temporary subset before the full CPU extraction. The final run must use the complete splits and the pinned revision.

## Definition of Done

- All tests pass and repeated evaluation of the same cache returns identical Recall@K.
- Embeddings are finite CPU `float32`, have the expected shapes, and each row has L2 norm approximately 1.
- Labels and paths align one-to-one with embedding rows; cache hashes verify successfully.
- Retrieval excludes self-matches and evaluates exactly 5,924 queries against 5,924 gallery items.
- `metrics.json` contains enough secret-free configuration and provenance to reproduce the run.
- No dataset, model checkpoint, or generated output is committed.
