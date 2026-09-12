# E2A-M4 Implementation Plan — Attention Pooling

## Objective and Experimental Contract

Learn content-dependent weights over final-layer DINOv3 spatial patch tokens and use their weighted sum as a 768-dimensional retrieval embedding. Exclude CLS and every register token before attention, then L2-normalize the pooled vector.

M4 changes only patch aggregation. Keep the backbone frozen and preserve the pinned model revision, processor, CUB class split, sample order, Proxy Anchor training protocol, leakage-free model selection, cosine evaluator, self-match exclusion, tie policy, Recall@K, seed, and CUDA-first device policy. Do not add CLS fusion, output projection, uncertainty, reranking, or backbone fine-tuning.

## Attention Architecture

Implement a minimal additive scorer shared across all patches:

```text
patch [B, P, 768]
  → Linear(768, 128)
  → Tanh
  → Linear(128, 1)
  → softmax over P
  → weighted sum of original patch vectors
  → L2 normalization
  → embedding [B, 768]
```

Initialize linear layers with Xavier-uniform weights and zero biases. Compute logits, softmax, weighted sum, and normalization in float32. Attention weights must be finite, non-negative, and sum to one per image. Do not add entropy regularization in the canonical run; record mean entropy and maximum weight as diagnostics only.

Train the scorer with the existing device-agnostic Proxy Anchor loss: 100 train-class proxies, `alpha=32`, `margin=0.1`, AdamW, scorer learning rate `1e-4`, proxy learning rate `1e-2`, weight decay `1e-4`, and the same scheduler/balanced batch settings as M3. This isolates pooling as the experimental variable.

## Configuration

Add `configs/cub_e2a_m4.yaml` extending the base config:

```yaml
representation: attention_pool
output_dir: outputs/attention_pool
attention_hidden_dim: 128
patch_cache:
  dtype: float32
  shard_size: 128
  max_cached_shards: 4
  prefetch_factor: 2
training:
  loss: proxy_anchor
  epochs: 30
  classes_per_batch: 16
  samples_per_class: 4
  validation_fraction: 0.1
  alpha: 32.0
  margin: 0.1
  attention_lr: 0.0001
  proxy_lr: 0.01
  weight_decay: 0.0001
  scheduler_step: 10
  scheduler_gamma: 0.5
```

Validate every M4 key without changing valid M1–M3 configs. Keep float32 as the canonical cache dtype; any float16-cache experiment is a separately named ablation. Resolve `device: auto` once at startup and record the actual extraction and training devices.

## Work Packages

### WP1 — Patch extraction and sharded cache

- Add a raw patch helper that slices `last_hidden_state[:, 1 + num_register_tokens:, :]`; never hard-code register or patch counts.
- Validate rank, positive patch count, hidden dimension, fixed token layout, and finite values.
- Extract the frozen backbone once for train and eval using CUDA whenever available. Keep JPEG decode and preprocessing in CPU workers, then use pinned batches and non-blocking CPU→CUDA transfers for inference.
- Write CPU float32 patch tensors in atomic fixed-size shards under `outputs/attention_pool/cache/{train,test}/`.
- Create a manifest containing shard paths, row ranges, labels, image paths, shapes, dtype, model revision, processor, token layout, config, per-shard SHA-256, and total count.
- On reuse, verify every checksum and all provenance fields before training. Never reuse normalized M1/M2 embeddings or M3 raw CLS/Mean cache.

### WP2 — Lazy cache dataset

- Implement a dataset mapping each global sample index to its shard and local row.
- Load shards lazily with a bounded LRU cache; do not concatenate all patch tokens in RAM.
- Keep labels and paths in manifest order and support the existing class-balanced sampler.
- Ensure DataLoader workers do not duplicate an unbounded cache. With CUDA, use `pin_memory=True`, `persistent_workers=True` for multi-epoch loaders, configured `prefetch_factor`, and `non_blocking=True` transfers. Only set persistence/prefetch when `num_workers > 0`.
- Never construct CUDA tensors inside Dataset/DataLoader workers. Workers handle disk I/O, JPEG decoding, preprocessing, and shard reads on CPU; the main training process owns CUDA.
- Define deterministic shard ordering and reject missing, overlapping, duplicated, or out-of-range row spans.

### WP3 — Attention model and shared training

- Add `AttentionPool` to `src/representations.py`, returning embeddings and optionally attention weights for diagnostics.
- Reuse `ProxyAnchorLoss`-style training utilities, balanced batches, seeded setup, gradient clipping, AdamW, and StepLR from M3. Move the attention scorer, proxies, loss inputs, validation embeddings, and compute-heavy metrics to CUDA whenever available.
- Generalize shared learned-representation utilities instead of copying Proxy Anchor, split, checkpoint, or device-transfer logic.
- Assert backbone parameters remain frozen and only scorer/proxy parameters receive gradients. Add runtime assertions that model parameters and every training batch share the resolved device.
- Guard against NaN/Inf, zero pooled vectors, empty patch sequences, incompatible checkpoint dimensions, and softmax along the wrong axis.

### WP4 — Leakage-free selection and final training

- Reuse the seeded per-class 90/10 split within classes 0–99.
- Train candidate epochs only on the internal training subset.
- Select epoch by validation Recall@1; break ties by lower validation loss, then earlier epoch.
- Reinitialize scorer and proxies, then train on all 5,864 training images for exactly the selected epoch count.
- Do not load or inspect the 5,924 evaluation labels/features during epoch selection. Evaluate them once after final training.

### WP5 — Checkpoints, extraction, and orchestration

- Add `scripts/train_attention_pool.py` with `--smoke-test`, `--overwrite`, and provenance-checked resume support.
- Extend `scripts/run_e2a.py` to route `attention_pool` through: patch cache → selection → final retraining → embedding export → shared evaluation.
- Save selection/final checkpoints atomically with scorer/proxy/optimizer/scheduler states, selected epoch, architecture, source-cache manifest hash, config, seed, and training history.
- Update generic representation validation so the shared evaluator accepts M4 metadata without weakening M1–M3 checks.

### WP6 — Artifacts and diagnostics

Produce:

```text
outputs/attention_pool/
├── cache/
│   ├── train/
│   ├── test/
│   └── manifest.json
├── checkpoints/
│   ├── selection_best.pt
│   └── final.pt
├── train_embeddings.pt
├── test_embeddings.pt
├── labels.pt
├── attention_diagnostics.json
├── training_history.json
└── metrics.json
```

Final embeddings must be finite CPU float32 tensors with shapes `[5864, 768]` and `[5924, 768]`, row norms approximately one, and exact label/path alignment. Diagnostics should summarize attention entropy, maximum weight, and effective attended-patch count by split without storing all attention maps.

### WP7 — Tests and comparison

- Test exact exclusion of CLS/register tokens using sentinel values.
- Test attention shape, softmax axis, weight sum, non-negativity, output dimension/norm, deterministic initialization, and gradient flow only through scorer/proxies.
- Verify uniform logits reproduce Mean Patch pooling exactly; this is the key M2→M4 sanity check.
- Test single/multiple patches, invalid register counts, empty sequences, NaN/Inf, extreme logits, and batch size one.
- Test shard boundaries, LRU eviction, checksums, corrupt manifests, balanced random access, and deterministic ordering.
- Test DataLoader guards: pinned memory on CUDA, persistence/prefetch only with positive workers, non-blocking transfer, and valid CPU fallback.
- Run synthetic smoke training and checkpoint reload; require finite decreasing loss and identical embeddings after reload.
- Keep all M1–M3 tests passing.
- After the full run, report M1–M4 Recall@1/2/4/8, dimension, trainable parameter count, cache size/dtype, training time, selected epoch, and attention diagnostics.

## Verification Sequence

During implementation, run only synthetic checks:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/train_attention_pool.py \
  --config configs/cub_e2a_m4.yaml \
  --smoke-test
```

The later full run is:

```bash
.venv/bin/python scripts/run_e2a.py \
  --config configs/cub_e2a_m4.yaml
```

Use `--overwrite` only for intentional replacement. Cache reuse/resume must fail closed when model revision, processor, token layout, config, or hashes differ.

## Definition of Done

- CLS/register tokens cannot influence M4 output.
- Uniform attention numerically matches M2 Mean Patch.
- Only attention scorer and training proxies are optimized.
- With CUDA available, backbone inference, attention training, validation embeddings, Proxy Anchor, and retrieval run on CUDA while CPU workers keep the input pipeline supplied through pinning and prefetch.
- Validation selection never accesses evaluation classes.
- Patch cache is sharded, bounded in RAM, checksum-verified, and reproducibly ordered.
- Final M4 artifacts satisfy the same retrieval schema and evaluator semantics as M1–M3.
- Re-evaluating the same cache produces identical Recall@K.
- No M1–M3 config, cache, checkpoint, embedding, or metric is overwritten.
