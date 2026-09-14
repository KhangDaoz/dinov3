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
- the same validation split and test-selection split (classes 100--199).

Only representation construction may differ. Learned pipelines must record
their objective, initialization, optimizer, checkpoint rule, parameter count,
and validation-selected hyperparameters. Fixed pipelines must not receive an
artificial training stage.

## Test-based selection protocol

Validation remains responsible for checkpoint and epoch selection within M3
and M4. After all four pipelines pass validation acceptance, evaluate M1--M4
on classes 100--199 and select the representation from those test results.
This is therefore a **test selection split**, not an untouched final test.
Record the resulting selection bias explicitly in every E2A report.

E1-R0 may be compared with M1 descriptively on the same split, but it is not
part of the M1--M4 winner rule.

## Winner rule

Select the pipeline lexicographically by integer test counts: highest
Hits@1, then Hits@2, Hits@4, and Hits@8. If all hit counts tie, prefer fewer
total optimized parameters; if complexity also ties, use the fixed order M1,
M2, M3, M4. Recall values use the common 1,177-query denominator for reporting
and are not compared with a floating-point tolerance.

Recall@2/4/8 are secondary tie breakers, not a composite score.

## Shared artifacts

Every pipeline must save:

- raw or manifest-addressed development/test embeddings with provenance;
- validation and test-selection Recall@1/2/4/8;
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
5. Evaluate M1--M4 on test classes 100--199 and save Top-100 artifacts.
6. Apply the registered rule to test Hits@K and write
   `outputs/e2a_selection/test_selection.json`; optionally compare M1 with
   E1-R0 descriptively.
7. Export the test-selected representation and Top-100 artifacts to E2B.

Because selection uses test performance, E2A has no remaining independent
holdout for an unbiased final generalization estimate.

Run the complete test-selection stage after accepting all validation outputs:

```bash
python scripts/run_e2a_test_selection.py --configs \
  configs/cub_e2a_m1.yaml configs/cub_e2a_m2.yaml \
  configs/cub_e2a_m3.yaml configs/cub_e2a_m4.yaml
```

The command evaluates every method, saves each `metrics/test.json` and
`rankings/test_top100.pt`, verifies common test IDs and recomputed integer
Hits@K, then writes `outputs/e2a_selection/test_selection.json`.
