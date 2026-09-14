# E2A Representation Pipeline Comparison Plan

## Scope

E2A compares four **representation pipelines** built on the same frozen
DINOv3 ViT-B/16 backbone. The comparison is not a claim that all four methods
are trainable models:

| Pipeline | Representation | Trainable component |
|---|---|---|
| M1 | Final CLS token | None |
| M2 | Mean of final patch tokens | None |
| M3 | CLS + mean-patch fusion followed by projection | Projection |
| M4 | Learned attention pooling over patch tokens | Attention aggregation |

M1 and M2 are deterministic transformations of frozen tokens. M3 and M4 may
learn aggregation parameters on development-fit data. All methods must emit
one L2-normalized image embedding and use the same cosine retrieval evaluator.
Register tokens are never included in patch pooling.

## Controlled comparison

All pipelines must share:

- the identical CUB class split and canonical seed-42 development
  fit/validation image IDs;
- the same pinned DINOv3 checkpoint, processor, source layer, image IDs, and
  token cache;
- frozen backbone parameters;
- the same self-match exclusion, stable tie policy, Recall@K implementation,
  and Top-100 ranking schema;
- the same validation labels and final-test isolation boundary.

Only representation construction may differ. Learned pipelines must record
their objective, initialization, optimizer, checkpoint rule, parameter count,
and validation-selected hyperparameters. Fixed pipelines must not receive an
artificial training stage.

## Test isolation

During implementation and tuning, run only validation evaluation for M1--M4.
Do not run, load, inspect, or reveal final-test metrics, rankings, or E1-R0
test values. When all four validation artifacts exist, create one immutable
selection lock containing method/config/cache/split hashes, Git commit,
validation metrics, and tie-rule trace. The final-test command must fail closed
when this lock is absent or inconsistent.

E1-R0 may be compared with M1 only after the lock opens the final-test stage.
It serves solely as an implementation regression check and cannot influence
the E2A winner.

## Winner rule

Select the pipeline lexicographically by integer validation counts: highest
Hits@1, then Hits@2, Hits@4, and Hits@8. If all hit counts tie, prefer fewer
total optimized parameters; if complexity also ties, use the fixed order M1,
M2, M3, M4. Recall values use the common 1,177-query denominator for reporting
and are not compared with a floating-point tolerance.

Do not replace this rule after observing results. Recall@2/4/8 are secondary
tie breakers, not a composite score, and test performance never participates
in selection.

## Shared artifacts

Every pipeline must save:

- raw or manifest-addressed development/test embeddings with provenance;
- validation and, after unlock, test Recall@1/2/4/8;
- per-query Top-100 candidate image IDs and cosine scores after self-match
  exclusion;
- query IDs, labels, split, config/cache/split hashes, schema version, command,
  Git state, environment, runtime, and trainable parameter count.

Top-100 is the shared downstream contract for E2B pair construction,
reranking experiments, and failure analysis. Full square similarity matrices
are not required.

## Execution order

1. Implement and accept M1 validation.
2. Implement and accept M2 validation using the frozen shared token cache.
3. Train/select M3 only on development fit/validation data.
4. Train/select M4 only on development fit/validation data.
5. Apply the winner rule and write the immutable selection lock.
6. Open final test once and evaluate M1--M4 for the complete comparison table;
   verify M1 against E1-R0. Test results cannot change the validation-selected
   winner.
7. Export the winning representation and Top-100 artifacts to E2B without
   changing the winning pipeline after test inspection.
