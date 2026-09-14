# E2A-M3 Implementation and Experiment Plan

## 1. Objective and method definition

E2A-M3 tests whether a supervised projection can fuse the accepted M1 global
CLS representation with the M2 patch summary. For image $x_i$:

\[
c_i=h_i^{\mathrm{CLS}},\qquad
p_i=\frac{1}{196}\sum_{j=1}^{196}h_{i,j}^{\mathrm{patch}},
\]

\[
v_i=W[c_i;p_i]+b,\qquad
z_i=\frac{v_i}{\lVert v_i\rVert_2},
\]

where $W\in\mathbb{R}^{768\times1536}$. M3 uses one affine projection
`Linear(1536, 768, bias=True)` and no hidden layer, activation, dropout,
uncertainty, or reranking. This isolates learned CLS/patch fusion from the
more expressive attention aggregation evaluated by M4.

The DINOv3 backbone remains frozen. Only the projection and training-only
metric-learning proxies are optimized. Initialize the projection with Xavier
uniform weights and zero bias using seed 42. Persisted retrieval embeddings
are FP32 and L2 normalization occurs immediately before metric loss or cosine
retrieval.

## 2. Learning objective

Train M3 with Proxy Anchor loss using one learnable proxy per development
class. Follow the published formulation with cosine-normalized embeddings and
proxies, scale \(\alpha=32\), and margin \(\delta=0.1\). Use the
[Proxy Anchor paper](https://arxiv.org/abs/2003.13911) and
[official PyTorch repository](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020)
as the method sources. Record the paper version, official repository commit
and license before implementation. Describe cached frozen DINOv3 inputs and
the lightweight fusion projection as a project adaptation, not an official
Proxy Anchor reproduction.

The 100 class proxies are auxiliary training parameters and are discarded at
inference. For the E2A complexity tie-break, count every optimized parameter:

- projection: \(768\times1536+768=1,180,416\);
- proxies: \(100\times768=76,800\);
- total trainable during fitting: 1,257,216;
- deployable representation parameters: 1,180,416.

This definition removes ambiguity in the “fewer trainable parameters” rule.

## 3. Locked data and selection protocol

Use the accepted M1/M2 cache artifacts as immutable inputs. Align them by
image ID rather than row position and verify identical labels, splits, model
revision, processor settings, CUB manifest hash, and `[11788, 768]` shape.
The M1 cache supplies CLS; the M2 cache supplies mean patch. Do not run the
backbone again for M3.

- Fit only on the canonical development-fit subset (about 80% per class,
  labels 0--99).
- Select checkpoints and report M3 only on the same 1,177 validation IDs used
  by M1/M2, with manifest hash
  `d9455ac8948ff640f94017339681a51744f6351f1db053732b3153a4c9c958ee`.
- M3 fitting and validation use different images but the same classes 0--99.
  The test selection split contains unseen classes 100--199. Therefore validation
  measures within-class-set image generalization and is an imperfect proxy
  for the class-generalization required by test selection; record this explicitly
  as a limitation.
- Seed 42 is the preregistered M3 decision run and must be used consistently
  for projection/proxy initialization, balanced sampling, and training. Do not
  choose among random seeds using validation performance.
- Select the epoch lexicographically by the integer validation hit counts:
  highest Hits@1, then Hits@2, Hits@4, Hits@8, then earliest epoch. Do not use
  a floating-point tolerance for epoch selection. Recall values are derived
  from these counts only for reporting.
- Do not tune architecture, loss, margin, scale, optimizer, or learning rate
  after observing M3 validation results.

M3 validation selects its checkpoint but does not authorize an E2A winner.
No test feature row may be passed through the learned projection during M3
training; test projection happens later in the common selection stage.

The later pipeline winner is selected by integer test Hits@1, Hits@2,
Hits@4, Hits@8, total optimized parameter count, then fixed M1--M4 order.
Recall values are not compared with a floating-point tolerance.

## 4. Fixed training recipe

Use a class-balanced global batch with 20 classes and 4 images per class
(80 samples globally; 40 per T4). Sample only development-fit IDs. Use one
process per T4 with DDP and a differentiable global gather before Proxy Anchor
loss so the loss sees the intended global batch; a nonlinear proxy loss must
not be computed independently on two unrelated local batches and merely
averaged.

- epochs: 30;
- optimizer: AdamW;
- projection/proxy learning rate: `1e-4`;
- weight decay: `1e-4` for projection weights, zero for bias and proxies;
- scheduler: cosine decay to zero, stepped once per optimizer update;
- precision: use FP32 for projection and Proxy Anchor training. Initial T4
  execution showed non-finite FP16 backward gradients from the combination of
  Proxy Anchor scale and AMP loss scaling; this stability correction does not
  change the architecture, data, loss equation, or selection rule;
- gradient clipping: global norm 5.0;
- checkpoint evaluation: after every epoch;
- deterministic seed: 42 for initialization, sampling, and all training RNGs.

Abort on non-finite inputs, embeddings, proxies, loss, or gradients. Save the
sampler state and training history needed to reproduce the selected epoch.
Because only seed 42 is run, report M3 as a single-seed result, never as
mean~\(\pm\)~standard deviation. State the absence of multi-seed variance as
an experimental limitation.

## 5. Implementation phases

### Phase A -- Configuration and representation

Add `configs/cub_e2a_m3.yaml`. Extend `config_e2a.py` with typed training and
Proxy Anchor sections and an exact M3 contract: `cls_mean_projection`, final
CLS plus final mean patch, output dimension 768, and L2 normalization. M1/M2
must remain loadable without artificial training fields. Reject attention,
register pooling, uncertainty, reranking, and test-stage tuning.

Add `CLSMeanPatchProjection` to `models/representations.py`. It accepts two
aligned `[B,768]` tensors or a typed fused-feature batch, concatenates them in
fixed order `CLS, mean_patch`, and returns `[B,768]`. Test that gradients reach
only projection parameters and input order cannot silently change.

Add a focused `ProxyAnchorLoss` implementation in `models/metric_learning.py`.
Cross-check its equation and a small numeric example against the cited paper
and official implementation without copying third-party source blindly.

### Phase B -- Paired cache dataset and DDP training

Add a dataset/adapter that strictly joins M1 and M2 caches by image ID,
validates shared provenance, and exposes only development-fit rows to the
trainer. Add a deterministic distributed class-balanced batch sampler; every
global batch must contain positive examples for each sampled class.

Implement `training/representation.py` and `scripts/train_e2a.py`. Save one
atomic checkpoint per new validation winner containing projection state,
optimizer/scheduler state, proxy state, epoch, config hash, both input cache
hashes, split hash, validation metrics, parameter counts, and Git/environment
metadata. Rank 0 alone writes artifacts after distributed synchronization.

### Phase C -- Embedding export and validation

Generalize `evaluate_e2a.py` so M3 loads the selected projection checkpoint,
joins M1/M2 caches, and exports method-owned embeddings. During this stage,
materialize only development-fit and validation embeddings. Test embeddings
may be generated only during the common test-selection stage.

Evaluate exact cosine retrieval on validation and save ordered Top-100
candidate IDs and scores. Recompute Recall@1/2/4/8 from the serialized ranking
and require exact agreement with `metrics/validation.json`.

### Phase D -- Orchestration and reporting

Extend `run_e2a.py` to dispatch M3 as `train -> export -> validation` while
retaining the M1/M2 extraction flow. A validation rerun may reuse a checkpoint
only when config, input-cache, split, and code provenance match; otherwise
fail explicitly or retrain under a new run directory.

Update `reports/e2a.tex` with M1--M3 validation results, the M3 selected epoch,
loss provenance, and both parameter counts. Report M3-minus-M1 and M3-minus-M2
deltas descriptively, but do not declare the E2A winner before M4 validation.
Label M3 explicitly as a single-seed result and include neither seed-wise
mean nor standard deviation.

## 6. Planned files

```text
configs/cub_e2a_m3.yaml
src/uncertainty_retrieval/config_e2a.py
src/uncertainty_retrieval/data/fused_cache.py
src/uncertainty_retrieval/models/representations.py
src/uncertainty_retrieval/models/metric_learning.py
src/uncertainty_retrieval/training/representation.py
scripts/train_e2a.py
scripts/evaluate_e2a.py
scripts/run_e2a.py
tests/unit/test_config_e2a.py
tests/unit/test_fused_cache.py
tests/unit/test_representations.py
tests/unit/test_metric_learning.py
tests/unit/test_representation_training.py
tests/integration/test_e2a_m3_smoke.py
reports/e2a.tex
```

Extend genuinely generic E2A modules instead of introducing parallel M3-only
evaluation logic. Keep accepted M1/M2 artifacts and schemas unchanged.

## 7. Artifact contract

```text
outputs/e2a_fusion/m3/seed_42/
├── config_resolved.yaml
├── environment.json
├── metadata.json
├── inputs/
│   ├── cls_manifest.json
│   └── mean_patch_manifest.json
├── split/
│   ├── fit_image_ids.pt
│   ├── validation_image_ids.pt
│   └── manifest_hash.txt
├── checkpoints/best.pt
├── training/history.json
├── embeddings/
│   ├── fit.pt
│   ├── validation.pt
│   └── manifest.json
├── rankings/validation_top100.pt
└── metrics/validation.json
```

The embedding manifest must identify both source-cache SHA-256 hashes,
projection checkpoint hash, concatenation order, normalization policy, shape,
dtype, IDs, labels, split, selected epoch, and parameter counts. Ranking
artifacts retain the common E2A `[1177,100]` schema and validation-only gallery.

## 8. Tests and acceptance criteria

Focused unit tests must cover:

- exact M3 config acceptance and forbidden/unknown settings;
- cache alignment under permuted row order and rejection of ID, label, split,
  processor, checkpoint, or manifest drift;
- exact `CLS, mean_patch` concatenation, output shape, initialization,
  gradients, FP32 normalization, and parameter counts;
- Proxy Anchor positive/negative terms against a hand-computed example,
  missing-positive classes, finite gradients, and invalid labels;
- class-balanced sampler composition, deterministic seed behavior, and strict
  exclusion of validation/test IDs from optimization;
- DDP global-batch loss semantics and rank-0-only checkpoint writing. In
  particular, compare the loss and projection/proxy gradients produced by the
  differentiable gather across two ranks against the same concatenated global
  batch computed on one device, within a declared numerical tolerance;
- epoch selection by integer Hits@1, Hits@2, Hits@4, Hits@8, then earliest
  epoch, including cases where rounded Recall values appear tied;
- checkpoint/cache/config hash mismatch rejection;
- Top-100 integrity and metric recomputation.

During implementation, run only the focused E2A-M3 unit tests. Leave the full
suite, integration smoke test, training run, and validation experiment to the
user.

M3 is accepted only when the selected checkpoint is reproducible from fit
data, no validation/test sample contributes a gradient, both source caches and
the validation hash match M1/M2, all saved embeddings/rankings pass integrity
checks, and no test artifact exists. Completion of M3 authorizes M4
implementation, not E2A winner selection.

## 9. Execution

After focused unit tests pass, run on two T4 GPUs:

```bash
python scripts/run_e2a.py \
  --config configs/cub_e2a_m3.yaml \
  --stage validation
```

Run `--stage test` only in the later common M1--M4 test-selection stage.
