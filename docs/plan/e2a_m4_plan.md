# E2A-M4 Implementation and Experiment Plan

## 1. Objective and representation

E2A-M4 learns content-dependent weights over final-layer patch-token
representations to form a global retrieval embedding while keeping the
DINOv3 backbone frozen. For the 196 patch tokens
$H_i\in\mathbb{R}^{196\times768}$:

\[
q_{i,j}=w_2^\top\tanh(W_1h_{i,j}+b_1)+b_2,
\qquad
a_{i,j}=\operatorname{softmax}_{j}(q_{i,j}),
\]

\[
v_i=\sum_{j=1}^{196}a_{i,j}h_{i,j},
\qquad
z_i=\frac{v_i}{\lVert v_i\rVert_2}.
\]

Use attention hidden dimension 256 with `Linear(768,256)`, `tanh`, and
`Linear(256,1)`. Do not add a value projection, output projection, CLS fusion,
multi-head attention, dropout, uncertainty, or reranking. CLS and all four
register tokens are excluded before attention. Softmax is only over the 196
patch positions, and each row of attention weights must sum to one.

Initialize both attention weights with Xavier uniform and biases with zero,
using seed 42. The deployable attention module has 197,121 parameters. The
100 training-only Proxy Anchor class proxies add 76,800 parameters, giving
273,921 total optimized parameters for the E2A complexity tie-break.

## 2. Controlled relationship to M1--M3

Keep the accepted protocol unchanged: pinned DINOv3 ViT-B/16 revision,
official 224-by-224 processor, fixed CUB manifests, exact cosine retrieval,
image-ID self-match exclusion, stable gallery-index ties, Recall@1/2/4/8, and
ordered Top-100 candidate IDs and scores.

M4 and M3 share Proxy Anchor, global batch construction, optimizer, learning
rate, scheduler, gradient clipping, and checkpoint-selection protocol so that
the learned aggregation is the principal change. Precision is specified
separately for each pipeline. Use scale \(\alpha=32\), margin \(\delta=0.1\),
class-balanced global batches, AdamW, and the same cited
[Proxy Anchor paper](https://arxiv.org/abs/2003.13911) and
[official implementation](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020).
M4 remains a DINOv3 adaptation rather than an official reproduction.

M1/M2 caches contain only `[11788,768]` global embeddings and cannot provide
the per-patch inputs required by M4. Do not reconstruct or repeat mean vectors
as fake patch tokens. Extract a dedicated development-only patch-token cache
from the identical frozen checkpoint and processor.

## 3. Test isolation and class-generalization limitation

During M4 development, extract only development classes 0--99. Split these
images with the canonical image-level 80/20 split: 4,687 fit and 1,177
validation images. Fit and validation therefore contain different images but
the same classes. Final test contains unseen classes 100--199, so validation
does not directly measure class-generalization; state this limitation in the
report.

No test patch feature, embedding, label, metric, or ranking may be generated,
loaded, or inspected before all M1--M4 validation artifacts are accepted and
the immutable selection lock is written. After unlock, run final test once for
all M1--M4 pipelines to create the complete comparison table; extract M4 test
patch tokens only at that point. Test results cannot change the winner already
selected from validation. Never use test tokens to select a checkpoint or
change the attention architecture.

Seed 42 is the single preregistered M4 run and must control initialization,
sampling, workers, and all training RNGs. Report a single-seed result, not
mean~\(\pm\)~standard deviation, and record missing multi-seed variance as a
limitation.

## 4. Patch-token cache

Extract DINOv3 output once with two T4 processes. Preserve token order
`CLS, register, patch`, validate four register and 196 patch tokens, then save
only final patch tokens. Backbone inference may use FP16 AMP; persist patch
tokens as contiguous CPU FP16 to limit storage. Cast each training batch to
FP32 before attention and Proxy Anchor computation.

The development tensor is approximately 1.8 GB. Save atomic rank shards
rather than constructing a second merged tensor in RAM:

```text
outputs/e2a_attention_pool/cache/development/
├── rank0.pt
├── rank1.pt
└── manifest.json
```

The manifest records shard hashes, image IDs, labels, split membership, shape,
dtype, patch/register counts, checkpoint revisions, processor settings, CUB
manifest hash, extraction command, Git state, and schema version. Load shards
with memory mapping where supported and join logically by image ID. Reject
duplicates, missing development IDs, test IDs, incorrect labels, non-finite
tokens, wrong dimensions, or provenance differing from accepted M1/M2.

The cache must contain exactly 5,864 development images shaped
`[5864,196,768]` across its shards. Do not create the final-test cache during
this phase.

Before accepting FP16 storage, run a deterministic fidelity check on a
class-stratified subset of 128 development images. Retain the backbone's FP32
patch output in memory, serialize the same tokens to FP16, reload and cast them
to FP32, then compare paired tensors. Require every image's relative L2 error
to be at most `1e-3`, minimum per-patch cosine similarity to be at least
`0.99999`, and cosine similarity between the FP32 and round-trip mean-patch
embeddings to be at least `0.99999`. Save subset IDs, summary statistics, and
thresholds in the cache manifest. Reject FP16 caching if any threshold fails;
fall back to FP32 shards rather than weakening thresholds after inspection.

## 5. Training and checkpoint selection

Train only the attention scorer and Proxy Anchor proxies. Use the M3 recipe:

- global batch: 20 classes by 4 images = 80; local batch 40 per T4;
- DDP: one process per T4 with differentiable global embedding gather;
- epochs: 30;
- optimizer: AdamW, learning rate `1e-4`;
- weight decay: `1e-4` for attention weights, zero for biases and proxies;
- scheduler: cosine decay to zero per optimizer update;
- precision: FP32 attention, normalization, cosine logits, log-sum-exp, loss,
  and backward;
- gradient clipping: global norm 5.0;
- checkpoint evaluation: every epoch;
- deterministic seed: 42 everywhere.

Select the checkpoint lexicographically by integer validation counts:
Hits@1, then Hits@2, Hits@4, Hits@8, then earliest epoch. Do not use a
floating-point tolerance for epoch selection. The E2A pipeline winner rule
uses the same integer ordering: validation Hits@1, Hits@2, Hits@4, Hits@8,
total optimized parameter count, then fixed M1--M4 order.

Use the same tested DDP global-gather semantics as M3. Add an M4-specific
single-device versus two-rank test comparing loss and gradients for attention
weights and proxies on the identical global batch.

## 6. Implementation phases

### Phase A -- Configuration and model

Add `configs/cub_e2a_m4.yaml` and an exact M4 contract in `config_e2a.py`:
method `m4`, representation `attention_pool`, source `final_patch`, attention
hidden dimension 256, output dimension 768, and FP32 training. Keep M1--M3
configs backward compatible and reject CLS fusion, register pooling,
projection heads, uncertainty, and reranking.

Add `AttentionPatchPooling` to `models/representations.py`. Return both the
pooled embedding and attention weights through a typed output. Validate input
shape, finite values, normalized attention weights, output shape, and zero
access to CLS/register fields.

### Phase B -- Development patch extraction

Add `data/patch_token_cache.py` and extend `extract_e2a_features.py` or add a
focused patch-token extraction mode. The extraction dataset must be filtered
to `record.split == "development"` before image decoding. DDP shards must use
image ID as the unique merge key and must not contain sampler-padding
duplicates in the final logical manifest.

Cross-check the checkpoint, processor, image IDs, labels, and CUB hash against
accepted M1/M2 manifests. Cache reuse is all-or-nothing; partial or stale
shards trigger explicit rejection rather than silent mixing.

### Phase C -- Training

Generalize `M3TrainingModel` and `training/representation.py` into reusable
learned-representation training without changing M3 behavior. Reuse the
class-balanced sampler, global differentiable gather, integer Hits checkpoint
selection, atomic checkpointing, and provenance checks.

Save attention state, training-only proxies, optimizer/scheduler state,
selected epoch, integer validation hits, Recall@K, cache/config/split hashes,
parameter counts, and environment in `checkpoints/best.pt`. Rank 0 alone
writes after a distributed barrier.

### Phase D -- Validation export and reporting

Extend `evaluate_e2a.py` to load the selected attention checkpoint and export
FP32 fit/validation embeddings. Also store validation attention weights as
FP16 plus per-image attention entropy for diagnostic and interpretability
analysis. These weights are not evidence of object localization, part
localization, or causal feature importance, and they never alter ranking.

Save exact validation cosine Top-100 and recompute Recall@K from its serialized
candidate IDs. Update `reports/e2a.tex` with all M1--M4 validation rows,
M4-minus-M1 and M4-minus-M3 deltas, selected epoch, parameter counts,
single-seed qualification, and class-generalization limitation. Only after
artifact acceptance may the registered winner-selection command create the
lock; do not declare a winner inside the M4 trainer or evaluator.

## 7. Planned files

```text
configs/cub_e2a_m4.yaml
src/uncertainty_retrieval/config_e2a.py
src/uncertainty_retrieval/data/patch_token_cache.py
src/uncertainty_retrieval/models/representations.py
src/uncertainty_retrieval/training/representation.py
scripts/extract_e2a_features.py
scripts/train_e2a.py
scripts/evaluate_e2a.py
scripts/run_e2a.py
tests/unit/test_config_e2a.py
tests/unit/test_patch_token_cache.py
tests/unit/test_representations.py
tests/unit/test_representation_training.py
tests/unit/test_m4_ddp.py
tests/integration/test_e2a_m4_smoke.py
reports/e2a.tex
```

Prefer extending shared E2A code where semantics are identical. Preserve all
accepted M1--M3 artifacts and schemas.

## 8. Artifact contract

```text
outputs/e2a_attention_pool/m4/seed_42/
├── config_resolved.yaml
├── environment.json
├── metadata.json
├── inputs/patch_token_manifest.json
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
├── attention/
│   ├── validation_weights.pt
│   └── validation_entropy.pt
├── rankings/validation_top100.pt
└── metrics/validation.json
```

No path containing `test` may exist before selection unlock. Top-100 must have
shape `[1177,100]`, contain only validation candidates, exclude self-match,
store finite descending cosine scores, and reproduce metrics JSON exactly.

## 9. Focused tests and acceptance

Unit tests must cover:

- strict M4 config and forbidden field combinations;
- attention output/weight shapes, softmax sums, finite FP32 behavior, exact
  exclusion of CLS/register tokens, gradients, initialization, and parameter
  counts;
- patch cache sharding, memory-mapped loading, ID-order reconstruction,
  hashes, development-only enforcement, corruption/provenance failures, and
  the preregistered FP16 round-trip fidelity thresholds;
- fit/validation membership with zero overlap and no unseen-class row;
- class-balanced sampling and integer Hits epoch tie rule;
- Proxy Anchor loss and gradient equivalence between two-rank differentiable
  gather and the identical single-device global batch for attention/proxies;
- checkpoint and cache hash mismatch rejection;
- attention entropy bounds, Top-100 integrity, metric recomputation, and
  final-test fail-closed behavior.

During implementation, run only focused M4 unit tests. Leave the complete
suite, DDP/integration smoke test, patch extraction, and training experiment
to the user.

Accept M4 only when all cache shards and checkpoint provenance validate, the
selected epoch follows integer Hits ordering, no validation/test row supplied
a training gradient, M1--M4 use identical validation IDs, all embeddings and
rankings pass integrity checks, and no final-test artifact exists. M4
acceptance completes validation comparison but does not itself open final
test; winner selection and lock creation are a separate audited step.

## 10. Execution

After focused tests pass, run on two T4 GPUs:

```bash
python scripts/run_e2a.py \
  --config configs/cub_e2a_m4.yaml \
  --stage validation
```

The runner performs development patch extraction (or validates an existing
cache), M4 training, embedding/attention export, and validation retrieval. It
must reject `--stage test` until the immutable M1--M4 selection lock exists.
