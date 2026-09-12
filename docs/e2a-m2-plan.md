# E2A-M2 Implementation Plan — Mean Patch Pooling

## Objective and Experimental Contract

Implement a frozen DINOv3 ViT-B/16 Mean Patch baseline. For each final-layer token tensor, exclude the CLS token and all register tokens, average only spatial patch tokens, then L2-normalize the resulting 768-dimensional vector. Evaluate cosine Recall@1/2/4/8 with the same CUB-200-2011 protocol as M1.

M2 changes only representation pooling. Backbone revision, processor, image ordering, class-disjoint split, retrieval code, self-match exclusion, tie policy, and seed must remain identical to M1. M2 performs no training, projection, uncertainty estimation, or reranking.

## Preconditions

1. Preserve `configs/cub_e2a.yaml` and `outputs/cls/` as the M1 configuration and artifacts.
2. Add `configs/cub_e2a_m2.yaml` by copying M1 settings and changing only:

   ```yaml
   representation: mean_patch
   output_dir: outputs/mean_patch
   ```

3. Use Python 3.14.7 from `.venv` and `device: auto`, preferring CUDA when available.
4. Keep the pinned model revision and dataset counts: 5,864 train and 5,924 eval images.

## Implementation Work Packages

### WP1 — Generalize representation selection

- Allow exactly `cls` and `mean_patch` in config validation.
- Add a central representation dispatcher so extraction does not branch across scripts.
- Derive output directory, metadata name, and validation messages from the resolved representation; remove M1-only CLI wording.
- Keep existing M1 behavior and artifact compatibility unchanged.

### WP2 — Implement Mean Patch pooling

- Add `mean_patch_embedding(last_hidden_state, num_register_tokens)` in `src/representations.py`.
- Slice tokens as `last_hidden_state[:, 1 + num_register_tokens:, :]`.
- Validate rank, non-negative register count, at least one remaining patch, finite input, and non-zero pooled vectors.
- Compute the mean in float32, normalize across the embedding dimension, and move only the persisted result to CPU.
- Read `num_register_tokens` from `model.config`; do not hard-code the DINOv3 register count.

### WP3 — Integrate extraction and metadata

- Pass model token-layout metadata to the dispatcher from `scripts/extract_features.py`.
- Continue producing CPU float32 embeddings with shapes `[5864, 768]` and `[5924, 768]`.
- Write M2 artifacts only under `outputs/mean_patch/` using the existing atomic-save and checksum flow.
- Record `representation: mean_patch`, register-token count, patch-token count, pooling rule, embedding dimension, processor, model revision, device, dtype, counts, and timings.
- Reject a cache whose metadata representation or config does not match the requested M2 run.

### WP4 — Reuse retrieval and evaluation

- Reuse the M1 cosine evaluator without changing ranking semantics.
- Replace representation-specific shape assumptions with metadata-backed validation while retaining expected CUB counts and dimension 768.
- Resolve GPU/CPU through the existing device policy and store the actual evaluation device.
- Ensure evaluation never reads from or modifies `outputs/cls/`.

### WP5 — Tests and comparison

- Add synthetic tests proving CLS and register tokens cannot affect Mean Patch output.
- Test zero, one, and multiple register tokens; malformed ranks; excessive/negative register counts; empty patch sets; NaN/Inf; zero vectors; dtype and L2 norm.
- Parameterize shared extraction/config/cache tests across `cls` and `mean_patch` to prevent M1 regressions.
- Verify retrieval results are invariant to chunk size using M2 embeddings.
- After the full run, add an M1-versus-M2 table containing representation, dimension, preprocessing, sample counts, extraction device/time, and Recall@1/2/4/8.

## Verification Sequence

Run only synthetic verification during implementation:

```bash
.venv/bin/python -m pytest -q
```

Then perform the M2 smoke and full runs explicitly with its config:

```bash
.venv/bin/python src/dinov3_backbone.py --config configs/cub_e2a_m2.yaml --root data
.venv/bin/python scripts/extract_features.py --config configs/cub_e2a_m2.yaml
.venv/bin/python scripts/evaluate.py --config configs/cub_e2a_m2.yaml
```

Alternatively, run both full stages with:

```bash
.venv/bin/python scripts/run_e2a.py --config configs/cub_e2a_m2.yaml
```

Use `--overwrite` only to intentionally replace existing M2 artifacts.

## Definition of Done

- M1's 14 existing unit tests still pass and new Mean Patch tests pass.
- Only patch tokens contribute to the pooled representation, verified by synthetic sentinel values.
- M2 embeddings are finite CPU float32 tensors of expected shape with row norms approximately one.
- Cache labels, paths, hashes, representation metadata, and row ordering are consistent.
- Evaluation processes exactly 5,924 queries/gallery items with self-match exclusion and deterministic ties.
- M1 and M2 differ only in representation logic and output location, making Recall@K directly comparable.
- No M1 artifact is overwritten and no uncertainty or trainable component enters M2.
