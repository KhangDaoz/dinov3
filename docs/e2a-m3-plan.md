# E2A-M3 Implementation Plan — CLS + Mean Patch Fusion

## Objective and Experimental Contract

Build a learned fusion representation from final-layer DINOv3 CLS and Mean Patch features. Concatenate the two raw 768-dimensional vectors into 1,536 dimensions, project once to 768 dimensions, and L2-normalize for cosine retrieval. Keep the DINOv3 backbone frozen.

M3 changes only feature fusion and its trainable projection. Preserve M1/M2 model revision, processor, CUB class split, image order, retrieval implementation, self-exclusion, tie policy, Recall@K, seed, and device policy. Do not add uncertainty, reranking, attention pooling, or backbone fine-tuning.

## Method and Training Decisions

Use a single `Linear(1536, 768, bias=True)` projection with Xavier-uniform weights and zero bias. Do not concatenate the separately normalized M1/M2 embeddings: cache raw float32 CLS and raw Mean Patch vectors, concatenate them, then project and normalize once.

Train with Proxy Anchor loss over the 100 training classes. Follow the authors' defaults `alpha=32`, `margin=0.1`, AdamW, weight decay `1e-4`, projection learning rate `1e-4`, and proxy learning rate `1e-2` (100×). The official implementation defines normalized embedding/proxy cosine similarity and these loss defaults in [losses.py](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020/blob/master/code/losses.py), and uses AdamW with the 100× proxy group in [train.py](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020/blob/master/code/train.py). Reimplement device-agnostically; do not copy its hard-coded CUDA utilities.

## Configuration

Add `configs/cub_e2a_m3.yaml`, extending the M1 base config and overriding:

```yaml
representation: fusion
output_dir: outputs/fusion
projection_dim: 768
training:
  loss: proxy_anchor
  epochs: 30
  classes_per_batch: 16
  samples_per_class: 4
  validation_fraction: 0.1
  alpha: 32.0
  margin: 0.1
  projection_lr: 0.0001
  proxy_lr: 0.01
  weight_decay: 0.0001
  scheduler_step: 10
  scheduler_gamma: 0.5
```

Validate all M3-only keys while leaving M1/M2 configs valid. `device: auto` must prefer CUDA.

## Work Packages

### WP1 — Raw frozen-feature cache

- Generalize config validation and representation dispatch to include `fusion`.
- Add helpers that return raw float32 CLS and raw Mean Patch features before L2 normalization.
- Extract the frozen backbone once for train and test; record register/patch counts, processor, revision, ordering, dtype, and SHA-256.
- Store private training cache under `outputs/fusion/cache/`; never reuse normalized `outputs/cls/` or `outputs/mean_patch/` as fusion input.
- Detect incomplete, stale, or mismatched feature caches before training.

### WP2 — Fusion model and Proxy Anchor loss

- Add a small `FusionProjection` module implementing concatenate → linear projection → L2 normalization.
- Implement Proxy Anchor with stable `logsumexp`/`log1p` operations, local labels 0–99, normalized proxies, no `.cuda()` hard-coding, and clear handling of classes absent from a batch.
- Add a class-balanced deterministic sampler with 16 classes × 4 images. Define epoch length and replacement behavior explicitly.
- Optimize only projection and proxies; assert every backbone parameter remains frozen.

### WP3 — Leakage-free model selection

- Make a seeded, per-class 90/10 train/validation image split inside classes 0–99; never inspect classes 100–199 during selection.
- Select the epoch by validation Recall@1, breaking ties by lower validation loss then earlier epoch.
- Retrain a freshly initialized projector on all 5,864 training images for exactly the selected epoch count.
- Evaluate the 5,924-image test split once after retraining. Store both selection and final-training histories.

### WP4 — Training, extraction, and orchestration

- Implement `src/train_projection.py` for model, loss, sampler, checkpoint schema, training, validation, and resume checks.
- Add `scripts/train_projection.py`; make extraction require the final M3 checkpoint before producing embeddings.
- Extend `scripts/run_e2a.py` for the ordered M3 flow: raw cache → select epoch → retrain → final embedding extraction → evaluation.
- Save checkpoints atomically with config, source-cache hashes, seed, epoch, optimizer/scheduler state, architecture, and loss hyperparameters.

### WP5 — Artifacts and evaluator compatibility

Produce the standard final artifacts:

```text
outputs/fusion/
├── cache/
├── checkpoints/
│   ├── selection_best.pt
│   └── final.pt
├── train_embeddings.pt
├── test_embeddings.pt
├── labels.pt
├── training_history.json
└── metrics.json
```

Final embeddings must be CPU float32 `[5864, 768]` and `[5924, 768]`, finite, row-normalized, and aligned with labels/paths. Extend cache validation from metadata instead of hard-coding a representation. Preserve M1/M2 outputs.

### WP6 — Tests and reporting

- Test raw feature slicing, concatenation order (`CLS` then Mean Patch), dimensions, initialization, normalization, frozen backbone, and gradient flow only through projection/proxies.
- Compare Proxy Anchor against a direct reference formula on tiny tensors; test absent-positive proxies, finite gradients, label range, and CPU/GPU device consistency where CUDA exists.
- Test balanced batches, deterministic validation split, checkpoint resume, stale-cache rejection, and selection tie-breaking.
- Keep all M1/M2 tests passing. Use synthetic smoke training to prove loss decreases and checkpoint reload reproduces embeddings.
- After the full run, report M1/M2/M3 dimension, trainable parameter count, training protocol/time, and Recall@1/2/4/8.

## Verification Sequence

During implementation, run only unit and synthetic smoke tests:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/train_projection.py --config configs/cub_e2a_m3.yaml --smoke-test
```

The later full run is:

```bash
.venv/bin/python scripts/run_e2a.py --config configs/cub_e2a_m3.yaml
```

Require explicit `--overwrite` or compatible `--resume` for existing M3 state.

## Definition of Done

- Backbone parameters never receive gradients; only projection and training proxies are learned.
- Validation/model selection never touches evaluation classes 100–199.
- Raw caches and checkpoints have verified provenance and deterministic ordering.
- Synthetic reference tests validate Proxy Anchor math and fusion behavior.
- Final embeddings and retrieval satisfy the same schema and evaluator semantics as M1/M2.
- A repeated evaluation of the same cache produces identical Recall@K.
- No M1/M2 artifact is modified, and the final report clearly separates selection, final training, and held-out test results.
