# E2A-M1 Implementation and Experiment Plan

## 1. Objective and scope

E2A-M1 establishes the frozen DINOv3 CLS-token retrieval baseline that all
later E2A representations must match or improve under an identical protocol.
For an image \(x_i\), M1 uses the final-layer CLS token directly:

\[
z_i = \frac{h^{\mathrm{CLS}}_i}
           {\lVert h^{\mathrm{CLS}}_i\rVert_2},
\qquad
s(i,j)=z_i^\top z_j.
\]

The embedding dimension is 768. M1 has no learned projection, classification
head, uncertainty, evidential embedding, reranking, or trainable parameter.
Register and patch tokens may be validated during extraction but are not part
of the M1 representation.

M1 is a DINOv3 off-the-shelf feature baseline, not a reproduction of a
third-party metric-learning method. Use the official
[DINOv3 paper](https://arxiv.org/abs/2508.10104) and
[ViT-B/16 model card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
as method/model sources. Record the DINOv3 license and the resolved checkpoint
revision.

## 2. Relationship to E1

E1 R0 and E2A-M1 have the same mathematical definition: frozen final CLS,
L2 normalization, cosine similarity, and self-match exclusion. E2A-M1 may
reuse the E1 CLS feature cache only after validating all provenance fields:

- model ID and resolved revision;
- final-layer CLS token mode and dimension 768;
- processor configuration and image resolution;
- register-token count;
- CUB manifest/split hashes, image IDs, labels, and record counts;
- feature dtype, schema version, and absence of NaN/Inf.

Do not use E1 evidence, \(\boldsymbol{\alpha}\), uncertainty, rankings, or
reranking outputs in E2A-M1. If cache provenance is absent or mismatched,
re-extract M1 features from the pinned DINOv3 checkpoint. The E1 R0 test
result must remain hidden during M1 development. It may be used only as a
regression oracle after the E2A winner is locked and the final-test stage is
explicitly opened; it is never a hyperparameter- or winner-selection signal.

## 3. Fixed data protocol

- CUB root: **data/CUB_200_2011/**.
- Development: original labels 0--99, expected 5,864 images.
- Final test: original labels 100--199, expected 5,924 images.
- Ignore **train_test_split.txt**; retain the class-disjoint split.
- Use one canonical stratified 80/20 development fit/validation split with
  seed 42. Persist image IDs and a split hash.
- Query and gallery are the same set within each evaluated split.
- Remove the query image by image ID before ranking.
- Use exact cosine ranking with stable gallery-index tie breaking.
- Report Recall@1, Recall@2, Recall@4, and Recall@8.

M1 has no fitting stage, but it must be evaluated on the canonical validation
split so M1--M4 can later be compared without consulting final-test labels.
Do not select an E2A winner until all four methods have validation results.
Do not run, load, inspect, or reveal any E2A final-test metric or ranking while
developing M1--M4. After all four validation runs are complete, apply the
prespecified winner rule, write an immutable selection-lock artifact, and only
then run or reveal the final-test comparison once. E1-R0 is checked against M1
only in this final-test stage.

Select the winner lexicographically by integer validation Hits@1, Hits@2,
Hits@4, and Hits@8. If all hit counts tie, prefer fewer total optimized
parameters; if still tied, use the fixed order M1, M2, M3, M4. Recall values
share the same 1,177-query denominator and are reported rather than compared
with a floating-point tolerance.

## 4. Reuse and required refactoring

Reuse the working E1 components where their semantics are method-independent:

- **data/cub.py**: manifests, fixed class split, validation split, count checks;
- **models/dinov3.py**: frozen model loading and token boundaries;
- **data/patch_cache.py**: portable CPU cache validation;
- **evaluation/retrieval.py**: L2 normalization, chunked cosine ranking,
  self-match masking, stable sorting, and Recall@K;
- **utils.py**: seeding, CUDA/DDP setup, cleanup, and environment metadata.

Do not route E2A through E1-specific config fields or uncertainty code.
Introduce a dedicated E2A configuration model. Keep existing E1 commands and
schemas backward compatible.

## 5. Implementation phases

### Phase A -- E2A configuration and representation contract

Add **configs/cub_e2a_m1.yaml** and an **E2AConfig** with strict unknown-key
rejection. It must define:

- dataset root, expected counts, validation fraction, and split seed;
- model ID, immutable revision, embedding dimension, register-token count,
  processor settings, and AMP extraction mode;
- representation name **cls**, source layer **final**, pooling **none**, and
  normalization **l2**;
- batch size, workers, prefetch factor, DDP world size, and similarity chunk;
- Recall@K values, self-match exclusion, stable tie policy, cache reuse policy,
  output root, and schema version.

Reject any M1 config containing projection, training, uncertainty, reranking,
patch pooling, or register-token pooling.

Add **models/representations.py** with a typed interface returning one
embedding per image. For mode **cls**, accept only a **DINOv3Tokens** object,
select **tokens.cls**, assert shape **[batch, 768]**, convert persisted
embeddings to FP32, and perform L2 normalization during retrieval rather than
mutating the raw cache.

### Phase B -- Provenance-safe feature preparation

Implement an E2A cache adapter that first checks a candidate E1/shared cache.
Cache acceptance must be all-or-nothing; never silently accept missing
metadata. Verify 11,788 unique image IDs, 5,864/5,924 split counts, label
alignment, shape **[11788, 768]**, finite features, model revision
**5931719e67bbdb9737e363e781fb0c67687896bc**, and final CLS token mode.

If validation fails, extract with one process per T4 via **torchrun** and merge
rank shards by image ID. Avoid duplicate model downloads when the Hugging Face
cache is already populated. Persist raw CPU FP32 CLS features once; downstream
evaluation transfers only chunks to the resolved CUDA device.

Export method-owned development and test embedding artifacts, or an immutable
manifest referencing the validated shared cache. Prefer a manifest plus hashes
to avoid copying identical 36 MB features unnecessarily, while retaining a
portable export option for E2B.

### Phase C -- M1 evaluation

Add an E2A evaluator that:

1. selects validation records by persisted split IDs;
2. verifies query/gallery label and image-ID alignment;
3. L2-normalizes features in FP32;
4. computes exact cosine similarities in GPU chunks;
5. masks self-matches by image ID, not diagonal position alone;
6. produces stable rankings and per-query hit vectors;
7. writes validation Recall@K without accessing final-test labels.

Provide a separate explicit **--stage test** command. It must refuse to run
unless an immutable E2A selection-lock artifact proves that validation for
M1--M4 is complete and records the winning pipeline, metrics, tie-rule
resolution, config hashes, and Git commit. There is no M1-only exception.

### Phase D -- Reporting and handoff

Create an M1 section/table in **reports/e2a.tex**, marked as incomplete until
M2--M4 are available. Save the exact M1 embedding manifest for future pair
sampling in E2B. The report must state that there is no variance over training
seeds because M1 contains no learned component.

## 6. Planned files

~~~text
configs/cub_e2a_m1.yaml
src/uncertainty_retrieval/config_e2a.py
src/uncertainty_retrieval/models/representations.py
src/uncertainty_retrieval/data/feature_cache.py
src/uncertainty_retrieval/evaluation/representation.py
scripts/extract_e2a_features.py
scripts/evaluate_e2a.py
scripts/run_e2a.py
tests/unit/test_config_e2a.py
tests/unit/test_representations.py
tests/unit/test_feature_cache.py
tests/unit/test_representation_evaluation.py
tests/integration/test_e2a_m1_smoke.py
reports/e2a.tex
~~~

Prefer extending a genuinely generic existing module over duplicating it.
Add a planned file only when it contains working behavior.

## 7. Artifact contract

~~~text
outputs/e2a_cls/<run_id>/
├── config_resolved.yaml
├── environment.json
├── metadata.json
├── split/
│   ├── validation_image_ids.pt
│   └── manifest_hash.txt
├── embeddings/
│   ├── manifest.json
│   ├── development.pt        # optional portable materialization
│   └── test.pt               # optional portable materialization
├── rankings/
│   ├── validation_top100.pt
│   └── test_top100.pt
└── metrics/
    ├── validation.json
    └── test.json
~~~

Embedding metadata must include image IDs, original labels, split, shape,
dtype, L2-norm policy, checkpoint/model revision, processor settings, source
layer/token, cache hash, CUB manifest hash, command, Git commit/dirty flag,
package versions, CUDA/cuDNN versions, GPU names, and wall time. Each ranking
artifact must store the ordered Top-100 candidate image IDs and corresponding
cosine scores for every query, after self-match exclusion. Top-100 supports
E2B pair construction, later reranking, and failure analysis while avoiding a
full \(n\times n\) ranking artifact. Also store query IDs, labels, split,
stable tie policy, and schema version.

## 8. Unit and integration tests

Unit tests must cover:

- strict M1 config validation and rejection of forbidden modes;
- CLS selection shape/dtype and exact exclusion of register/patch tokens;
- cache provenance match/mismatch, duplicate IDs, label drift, bad dimensions,
  non-finite features, and schema mismatch;
- L2 norms, stable cosine ties, image-ID self-match exclusion, split isolation,
  and hand-computed Recall@K;
- deterministic validation IDs/hash and artifact round-trip;
- confirmation that M1 has zero trainable parameters and never imports EDL or
  reranking code.

The mocked integration smoke test should emulate 1 CLS + 4 register + 196
patch tokens, execute two distributed extraction shards, merge them without
duplicates, evaluate a tiny retrieval fixture, and verify the output schema.
It must not download a real checkpoint.

## 9. Acceptance gates

M1 is ready for execution only when:

- source compilation and all unit tests pass;
- the CUB manifest has the fixed counts and no split overlap;
- the backbone is frozen and M1 exposes exactly **[N, 768]** CLS embeddings;
- single-GPU and 2-GPU extraction produce identical image-ID-aligned features
  within the declared numerical tolerance;
- repeated evaluation of the same cache produces bitwise-identical rankings;
- no test metric participates in configuration or winner selection.

Only after the E2A selection lock opens the final-test stage, E2A-M1 must
reproduce the accepted E1 R0 Recall@1/2/4/8 values within \(10^{-6}\). The
validation process must not load those E1 test values; the final-stage
regression checker reads them from the accepted E1 artifact only after unlock.

A mismatch blocks M1 acceptance and requires checking checkpoint revision,
processor, cache alignment, self-match masking, normalization, and tie policy.

## 10. Decision and handoff

E2A-M1 validation implementation is accepted when it passes all integrity
gates and establishes a reproducible validation reference. Test acceptance is
deferred until M1--M4 validation is complete and the selection lock opens the
final-test stage. Acceptance does not require improving E1 R0 because M1 is
the same baseline. Do not declare M1 the best E2A representation at this stage.

After M1 acceptance, freeze its config, split IDs, cache/manifest hashes,
evaluation code, and validation metrics. M2--M4 must reuse these exact
controls. Select the final E2A winner with the registered integer Hits@K,
total-optimized-parameter, and fixed-order rule before opening test or handing
its representation to E2B.
