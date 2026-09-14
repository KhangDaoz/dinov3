# E2A-M2 Implementation and Experiment Plan

## 1. Objective and hypothesis

E2A-M2 evaluates global average pooling over the final-layer DINOv3 patch
tokens as a fixed representation pipeline:

\[
v_i = \frac{1}{P}\sum_{p=1}^{P}h_{i,p}^{\mathrm{patch}},
\qquad
z_i = \frac{v_i}{\lVert v_i\rVert_2},
\qquad
s(i,j)=z_i^\top z_j.
\]

For 224-by-224 input and ViT-B/16, expect $P=196$ patch tokens and a
768-dimensional image embedding. The mean must include every final-layer
patch token exactly once and must exclude CLS and all four register tokens.
M2 has no projection, learned parameters, uncertainty, or reranking.

The scientific question is whether distributed local information in patch
tokens provides stronger fine-grained retrieval than M1 final CLS. This is a
comparison of representation pipelines, not a training comparison.

## 2. Locked experimental protocol

Reuse the accepted M1 controls without modification:

- frozen `facebook/dinov3-vitb16-pretrain-lvd1689m` at revision
  `5931719e67bbdb9737e363e781fb0c67687896bc`;
- official 224-by-224 processor settings recorded in the cache manifest;
- CUB labels 0--99 as development and 100--199 as final test;
- canonical stratified development validation IDs at fraction 0.2, seed 42;
- exact FP32 cosine retrieval, image-ID self-match exclusion, stable gallery
  index tie breaking, and Recall@1/2/4/8;
- integer validation Hits@1 as the primary E2A winner criterion, followed by
  Hits@2, Hits@4, Hits@8, total optimized parameter count, then fixed M1--M4
  order;
- ordered Top-100 candidate IDs and cosine scores for every query.

Only run M2 validation now. Do not run, load, inspect, or reveal M2 test
metrics or rankings. `--stage test` must continue to fail closed until M1--M4
validation is complete and the immutable selection lock exists. M1's accepted
validation result is a comparison row, not a hyperparameter signal; M2 has no
hyperparameter to tune.

## 3. Feature preparation and cache policy

The existing M1 cache contains only CLS embeddings, so it cannot produce M2.
Do not claim that it is a shared patch-token cache. Re-run the frozen backbone
with the identical checkpoint and processor, compute patch means on the GPU,
cast the resulting embeddings to CPU FP32, and save only `[11788, 768]`.
Avoid persisting all `[11788, 196, 768]` patch tokens because they are not
required by M2 and would create a multi-gigabyte artifact.

Create an M2-owned cache at
`outputs/e2a_mean_patch/cache/dinov3_mean_patch.pt`. Its metadata must include:

- model ID, requested and resolved immutable revision;
- processor settings and CUB manifest hash;
- source layer `final`, token source `patch`, pooling `mean`;
- expected patch count 196, register-token count 4, embedding dimension 768;
- schema version, dtype, image IDs, original labels, and development/test
  split membership.

Cache reuse is all-or-nothing. Reject missing metadata, wrong pooling/token
mode, non-finite values, duplicate IDs, shape/count drift, or provenance that
differs from M1. Never fall back to the M1 CLS cache as an M2 feature source.

## 4. Implementation phases

### Phase A -- Generalize the representation contract

Add `configs/cub_e2a_m2.yaml` with method `m2`, name `mean_patch`, source layer
`final`, pooling `mean`, normalization `l2`, expected patch count 196, and
output root `outputs/e2a_mean_patch/m2`.

Refactor `config_e2a.py` from an M1-only validator into method-specific M1/M2
contracts while preserving M1 config behavior. Reject register pooling,
CLS-plus-patch fusion, projection, attention, training, uncertainty, and
reranking fields in M2.

Extend `models/representations.py` with `MeanPatchRepresentation`. It must
accept `DINOv3Tokens`, validate patch shape `[B, 196, 768]`, compute
`tokens.patches.float().mean(dim=1)`, return `[B, 768]` FP32, and contain zero
trainable parameters. L2 normalization remains inside retrieval evaluation,
not in the persisted raw cache.

### Phase B -- Generalize extraction and cache validation

Make `extract_e2a_features.py` dispatch explicitly by the configured method.
Keep one process per T4, distributed sampling, AMP backbone inference,
non-blocking transfers, and rank-shard merge by image ID. Perform the mean in
FP32 to avoid accumulation error under AMP.

Generalize `feature_cache.py` to validate the configured representation rather
than naming all logic M1. Representation-specific provenance must prevent CLS
and mean-patch caches from being interchanged. Preserve existing M1 manifests
and schemas so the accepted M1 artifact remains readable and unchanged.

### Phase C -- Validation evaluation

Use the same persisted validation image IDs as M1 and verify their hash equals
`d9455ac8948ff640f94017339681a51744f6351f1db053732b3153a4c9c958ee`.
Fail before evaluation if the IDs, labels, query count (1,177), CUB manifest,
or processor/checkpoint provenance differs.

Reuse the generic chunked evaluator to produce Recall@1/2/4/8 and a Top-100
artifact. Recompute metrics from the saved ranking as an acceptance check.
Do not change M1 metrics and do not create a winner or selection lock yet.

### Phase D -- Reporting

Update the M2 row of `reports/e2a.tex` with validation results, parameter count
zero, cache hash, runtime, and an explicit note that training-seed variance is
not applicable. Keep all final-test cells hidden. Record the M2-minus-M1
Recall@K differences descriptively without declaring an E2A winner before M3
and M4 are validated.

## 5. Planned changes

```text
configs/cub_e2a_m2.yaml
src/uncertainty_retrieval/config_e2a.py
src/uncertainty_retrieval/models/representations.py
src/uncertainty_retrieval/data/feature_cache.py
scripts/extract_e2a_features.py
scripts/evaluate_e2a.py
scripts/run_e2a.py
tests/unit/test_config_e2a.py
tests/unit/test_representations.py
tests/unit/test_feature_cache.py
tests/unit/test_representation_evaluation.py
tests/integration/test_e2a_m2_smoke.py
reports/e2a.tex
```

Prefer extending the generic E2A entry points. Do not add M2-specific scripts
unless a method-independent entry point cannot express the behavior safely.

## 6. Artifact contract

```text
outputs/e2a_mean_patch/
├── cache/dinov3_mean_patch.pt
└── m2/
    ├── config_resolved.yaml
    ├── environment.json
    ├── metadata.json
    ├── split/
    │   ├── validation_image_ids.pt
    │   └── manifest_hash.txt
    ├── embeddings/manifest.json
    ├── rankings/validation_top100.pt
    └── metrics/validation.json
```

The ranking schema must remain identical to M1: 1,177 unique query IDs,
`candidate_image_ids` and `cosine_scores` shaped `[1177, 100]`, no self-match,
finite descending scores, and no candidate outside the evaluated validation
gallery.

## 7. Tests and verification

Unit tests must cover:

- acceptance of the exact M2 config and rejection of forbidden modes;
- hand-computed patch mean, FP32 accumulation/output, output shape, finite
  validation, zero parameters, and insensitivity to CLS/register values;
- rejection of zero/wrong patch counts and wrong embedding dimensions;
- M1/M2 cache cross-use rejection plus pooling, token-source, schema, ID,
  label, split, dtype, shape, and non-finite failures;
- equality of M1 and M2 validation IDs/hashes;
- Top-100 shape, score ordering, self-match exclusion, and metric
  recomputation.

Run only the focused E2A unit tests during implementation. Leave the full
suite, integration test, extraction, and validation experiment to the user.

## 8. Execution and acceptance

After focused unit tests pass, run M2 validation on 2-by-T4:

```bash
python scripts/run_e2a.py \
  --config configs/cub_e2a_m2.yaml \
  --stage validation
```

Accept M2 only when cache provenance is complete, the validation hash matches
M1, all binary-artifact integrity checks pass, metrics recomputed from the
saved Top-100 ranking exactly match JSON, and no test metric/ranking exists.
Completion of M2 authorizes implementation of M3, not winner selection or
final-test execution.
