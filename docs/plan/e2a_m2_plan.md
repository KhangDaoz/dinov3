# E2A-M2: mean patch representation plan

Protocol: e2ab-v2. Status: proposed implementation; no accepted metrics.

## 1. Method

Follow [the shared protocol](e2a_plan.md) and
[the new specification](../experiments/e2a-e2b-new.md).
For final patch tensor H_i with shape [196,768]:

~~~text
v_i = H_i.float().mean(dim=0)
z_i = v_i / ||v_i||_2
cosine(i,j) = z_i dot z_j
~~~

Include each patch once. Exclude CLS and all four register tokens.
Accumulate the mean in FP32; normalize the resulting vector, not individual
patches before averaging. M2 has zero trainable parameters and no projection.
It tests the patch summary against M1 under the same frozen feature pipeline.

## 2. Shared inputs and isolation

Use exactly the development patch cache supplying M4, with the same pinned
DINOv3, deterministic processor, token output semantics, precision, and
split manifest as M1. Source/model checks are in [sources](sources.md).
A CLS-only cache cannot provide M2; repeated mean vectors cannot provide M4.

The common cache has 5,864 development images; M2 exports [4687,768] Train
and [1177,768] Validation raw embeddings. Materializing patch means once
also provides the patch input for M3. Persist mean-cache hashes and source
patch-shard hashes so that M2/M3 agreement can be checked exactly.

Final Test classes 100–199 remain inaccessible to extraction and evaluation
until every E2A/E2B branch in the declared seed roster is locked.

## 3. Implementation steps

1. Add configs/cub_e2a_m2.yaml: method m2, source final_patch, pooling mean,
   patch count 196, register count 4, output dimension 768.
2. Extend MeanPatchRepresentation and strict config validation. Reject
   trainable heads, register pooling, pre-mean L2 normalization, and unknown
   modes.
3. Derive means in GPU batches from shared shards and persist a portable
   CPU FP32 cache or a manifest referencing immutable means.
4. Validate the exact Train/Validation IDs against the common manifest.
5. Evaluate validation using the shared scorer, saving Top-100 IDs/scores
   and integer Hits@1/2/4/8; rederive metrics from serialized rankings.
6. Freeze M2 inputs/config and export both development partitions to E2B-M2.
   Continue all remaining branches regardless of relative validation scores.

Extend the generic E2A modules used by M1 rather than duplicate evaluators.
Add tests/unit/test_representations.py coverage and
tests/integration/test_e2a_m2_smoke.py when implemented.

## 4. Artifacts and execution

Output root: outputs/e2a_mean_patch/<run_id>/seed_42/.

Required artifacts: config_resolved.yaml, environment.json, metadata.json,
source patch/mean/split manifests, embeddings/{train,validation}.pt or
references, rankings/validation_top100.pt, metrics/validation.json, freeze.json.
No optimizer or learned M2 checkpoint is required.

After implementation and the full test suite:

~~~bash
torchrun --standalone --nproc_per_node=2 scripts/run_e2a.py --config configs/cub_e2a_m2.yaml --stage validation
~~~

The joint final runner later creates [5924,768] test embeddings and evaluates
cosine, E2B pair-only, and fusion using the locked M2 branch. There is no
representation selection stage between E2A and E2B.

## 5. Tests and acceptance

Require a hand-computed mean fixture; FP32 accumulation; zero trainable
parameters; invariance to CLS/register changes and to patch permutation;
rejection of empty/wrong patch dimensions and nonfinite/zero pooled vectors.
Use a nonzero sentinel in register tokens to detect accidental inclusion.

Test source-hash mismatch, CLS/mean cache substitution, shuffled ID joins,
per-class split equivalence across M1–M4, and no test access before the global
lock. Verify the validation ranking shape [1177,100], membership, tie order,
self-match removal, metric denominator, and metric recomputation.

Require M2 mean tensors to equal the means used by M3 from the same source
patches. Fixed-feature training-seed variance is not applicable; report this
alongside parameter count zero. E2B-M2 has its own seeded learned pair head.
