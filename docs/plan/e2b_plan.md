# E2B: four-branch pair-wise confidence and fusion plan

Protocol: e2ab-v2. Status: proposed implementation; no accepted run or metrics.

## 1. Scope and experiment matrix

Follow [the new specification](../experiments/e2a-e2b-new.md),
[the shared protocol](e2a_plan.md), and [source lineage](sources.md).
Train independent pair heads for M1, M2, M3 and M4. Every E2A representation
is frozen before its pair head is trained. No earlier representation winner
or historical score is an input.

For each representation, report three primary Final Test modes:

| Mode | Candidate generation | Ranking |
|---|---|---|
| Cosine | Exact cosine over the split gallery | Original cosine order |
| Pairwise | Same representation's cosine Top-100 | Pair score, lambda=0 |
| Fusion | Same representation's cosine Top-100 | Validation-selected lambda |

This produces 12 primary rows per seed and three four-row result tables.
Pairwise means candidate-constrained reranking; it is not exhaustive learned
retrieval. Image-level evidential evaluation is outside the active scope.

## 2. Inputs and leakage barriers

Input z is a frozen L2-normalized FP32 768-vector from the corresponding
E2A freeze record. Verify model/processor/source-cache, split, config and
checkpoint hashes plus representation identity. M3/M4 pair heads use the
matching training-seed representation. Do not backpropagate into z.

Development/Train: 4,687 images in classes 0–99.
Development/Validation: 1,177 different images in those same classes.
Final Test: all 5,924 images in classes 100–199, available only after the
global lock for every branch and planned seed.

Training pair endpoints must both be in Train; validation endpoints both in
Validation. Reject self-pairs, mixed partitions and Final Test endpoints.
Mine candidates using only the current partition's features/gallery.
Validation rankings cannot construct training pairs.

A label-aware sampler/evaluator may create targets, but the pair scorer
accepts only features. Absolute class IDs, image IDs, rank positions,
labels and confidence targets are never network inputs.

## 3. Preserve the existing pair-sampling protocol

Per Train query and epoch, retain the original E2B recipe:

- 16 positive pairs, uniform over other same-class Train images, target 1.
- 8 hard-negative pairs, uniform over different-class candidates in its
  own representation's Train cosine Top-100, target 0.
- 8 random-negative pairs, uniform over different-class Train images,
  target 0.

Use sampling with replacement only when the eligible pool is too small;
record repeats. If the hard-negative pool is empty, fill from the random
negative pool. Missing valid positive or negative pools are errors.
There are 32 × 4,687 = 149,984 ordered pairs/epoch, half positive.

Mine the Train cosine Top-100 once per frozen representation and persist it.
Resample from the same static pool each epoch. Do not update hard negatives
from pair scores or fusion. Sort eligible image IDs for deterministic sampling.

Use keyed RNG streams (training_seed, epoch, query_id, pair_type) so positive
and random-negative draws can be matched across branches. Use the same
draw counts and seed policy for hard pools. The hard candidate identities
will differ because the representations differ; record this explicitly.
Training streams also specify a seeded 50% endpoint swap augmentation.
Inference always remains query-first; augmentation does not prove symmetry.

There is no common M1 mining pool imposed on other methods. This preserves
the existing representation-based mining rule and compares full pipelines.
The experiment does not isolate a head's behavior on identical pairs.
Report per-branch hard-pool statistics, pair repetition and candidate overlap;
a fixed-common-pair ablation would require a separate declaration.

Validation uses all 1,177 × 100 = 117,700 ordered cosine candidate pairs at
their natural match prevalence. Do not balance/resample the validation set.
Persist IDs, targets, cosine and prevalence independently for each branch.

## 4. Identical network and optimization policy

For normalized frozen a,b:

~~~text
u = concat(a, b, abs(a-b))                    # 2304 dimensions
Linear(2304,512) -> ReLU -> Dropout(0.1)
Linear(512,128)  -> ReLU -> Dropout(0.1)
Linear(128,1)                                # logit l
C(a,b) = sigmoid(l)                          # inference score
~~~

Every branch has 1,245,953 learned pair-head parameters. Use the same
initialization for a given seed: Linear Kaiming-uniform with a=sqrt(5),
bias uniform in [-1/sqrt(fan_in),1/sqrt(fan_in)].
Neither representation identity nor validation results change the architecture.

Training loss for pair target y is
BCE(l,y) = softplus(l) - y*l.
Use BCEWithLogitsLoss directly on logits, unweighted for the balanced pairs.
Do not apply sigmoid before the loss or claim the sigmoid is calibrated
under natural retrieval prevalence.

| Setting | All four pair branches |
|---|---|
| Epochs | 30 |
| Global batch | 4,096 pairs, 2,048 per T4 for full batches |
| Optimizer | AdamW lr 1e-4, betas (0.9,0.999), eps 1e-8 |
| Weight decay | 1e-4 on all pair-head parameters |
| Schedule | Cosine to zero over optimizer updates |
| Gradient clipping | Global norm 5.0 |
| Precision | FP32 head, BCE and backward |
| Training seed | Shared roster; primary 42 |
| Upstream representation | Frozen, identical split/cache controls |

One complete epoch uses every sampled pair exactly once in a seeded order.
There are 37 updates: 36 full batches and a final batch of 2,528 pairs
(1,264 per T4). Do not pad the production dataset or drop this tail.
For fixtures use explicitly weighted valid-sample masks if padding is needed.

Checkpoint criterion is minimum natural-candidate validation mean BCE, then
earliest epoch on an exact tie. Aggregate loss sums/counts globally in FP64;
validate with dropout disabled. Validation Recall is diagnostic at this
stage, not an alternative epoch-selection rule. The 30-epoch trajectory is
the fixed initial checkpoint search; other optimizer/architecture grids are
singleton. Any additional tuning requires the same declared budget and
search space in every branch and completion before the global lock.

Write model/optimizer/scheduler, RNG/sampler states, selected epoch,
validation loss, all provenance hashes and selection trace atomically.
Keep the selected Train-trained head; do not refit on Train+Validation.

## 5. Fusion and selection

For query q and candidate x in its cosine Top-N:

~~~text
score(q,x) = lambda * cosine(q,x) + (1-lambda) * C(q,x)
~~~

Cosine is in [-1,1], C in [0,1]. Preserve the raw formula without per-query
min-max scaling, temperature fitting or score transforms. Lambda is a score
weight, not a mixture of calibrated probabilities.

The primary candidate budget is N=100. Within that list, sort scores
descending and preserve original cosine order on exact ties.
Candidates outside Top-N keep baseline order and cannot be promoted.
At lambda=1 return the original cosine ranking directly to guarantee identity.

After freezing the BCE-selected checkpoint, select lambda independently per
branch/seed on N=100 Validation from {0,0.25,0.5,0.75,0.9,1}.
Choose maximum integer Hits@1, then Hits@2, Hits@4, Hits@8, then larger lambda.
Persist every validation grid value and decision trace. Lambda=1 or a
negative gain is a valid scientific result.

Preserve N in {10,20,50,100} as secondary candidate-budget diagnostics.
On Validation evaluate the full N/lambda grid from the same saved Top-100
pair scores. The primary lambda is still selected only at N=100.
On Final Test report primary N=100 cosine/pair/fusion tables plus
preregistered N diagnostics using lambda=0 and the frozen primary lambda.
Do not search a test lambda grid or retune lambda separately for test N.
No classification threshold is needed for ranking; any future threshold
must be fixed on Validation and declared before test.

## 6. Global lock and one final evaluation campaign

After E2A/E2B validation is complete for the declared seeds, create
global_lock.json with protocol version, timestamp, seed roster, common split,
backbone/processor/evaluator hashes, all four representation freeze records,
all pair-head selections, fixed N/lambda policy, reliability definitions,
statistical analysis plan and qualitative selection rules.
Include all checkpoint/config hashes and a SHA-256 of the lock payload.

The final runner refuses partial rosters, changed hashes, missing selection
traces, development overlap or unregistered settings. It never loads an
optimizer or trains. Process Final Test only after successful verification,
then score all planned methods/seeds in one campaign. Test labels enter
metric/diagnostic computation after scoring, not the scorer.

Save a final-access log. An interrupted campaign may resume identical frozen
work after validating hashes and existing outputs; no tuning is allowed.
A genuine implementation error requires a documented invalidation and
consistent recomputation of all affected rows. Prior exposure remains
disclosed; do not relabel a corrective rerun as an untouched test.

## 7. Reliability, statistics, and failure analysis

Primary endpoints are each of the four fusion-minus-matching-cosine
Recall@1 differences in percentage points. Report all four with negative
or zero outcomes. E2A cosine comparisons and pair-only effects are secondary.
Use [the reporting plan](research_verification.md) for seed summaries,
conditional uncertainty intervals and multiple-comparison limits.

Report Top-100 positive coverage:
fraction of queries with at least one positive in the candidate set.
For K<=100 it upper-bounds the achievable reranked Recall@K.
Different candidate coverage is part of the representation effect.

Reliability populations must remain separate:

1. All baseline Top-100 pairs: AUROC, average precision (define this as AP,
   not an unspecified PR-area integral), BCE, Brier score, 10 equal-width
   probability-bin ECE, reliability counts and positive prevalence.
2. Original cosine top-1 pair: target is that fixed candidate's class match.
   Compare cosine and its pair confidence via AUROC, AP and risk–coverage.
3. Optional post-fusion top-1: use its own candidate/target and clearly
   separate results from the original top-1 population.

Brier = mean((C-y)^2); ECE = sum_b (n_b/n)*|mean_b(C)-mean_b(y)|.
Bins are [0,0.1), …, [0.9,1]; empty bins contribute zero.
For risk–coverage sort confidence descending (image-ID ties), use risk
1 - mean correctness in each retained prefix, and AURC as the mean prefix
risk over coverages 1/n,…,1. Lower AURC is better. Undefined AUROC/AP cases
must carry null and a reason. Do not treat cosine as a probability for ECE
or Brier, and do not equate balanced training accuracy with retrieval recall.

Pair observations share images and galleries; do not use them as independent
samples for significance. These diagnostics do not establish universal
calibration, OOD detection or causality.

Before test, fix examples as the first 8 query IDs per representation/category:
wrong-to-correct, correct-to-wrong, still-wrong, still-correct (cosine vs fusion).
Export every category count plus IDs, scores and contact sheets where available.
Do not select only favorable examples. Final labels are permitted for this
registered analysis after all models are locked.

## 8. Implementation, CUDA and tests

Proposed files (create only when functional):

~~~text
configs/cub_e2b_{m1,m2,m3,m4}.yaml
src/uncertainty_retrieval/config_e2b.py
src/uncertainty_retrieval/data/pair_cache.py
src/uncertainty_retrieval/sampling/pairs.py
src/uncertainty_retrieval/models/pair_confidence.py
src/uncertainty_retrieval/training/pair_confidence.py
src/uncertainty_retrieval/evaluation/{pair_confidence,artifacts}.py
scripts/{prepare_e2b_pairs,train_e2b,evaluate_e2b,run_e2b}.py
scripts/{lock_e2ab,run_e2ab_final,verify_e2ab_outputs}.py
tests/unit/test_{config_e2b,pairs,pair_confidence,pair_reranking,e2b_artifacts}.py
tests/integration/test_e2b_smoke.py
~~~

Use both T4s with one process per GPU. BCE is additive, so DDP pair training
does not require embedding all-gather. Weight unequal rank counts correctly.
Disable dropout for two-rank gradient equivalence against the same single
global batch. Keep optimizer and head on the assigned GPU, and preload
small [N,768] feature matrices when memory permits.

Shard extraction/query/pair inference without padding; merge by query ID
and original candidate position. Rank 0 writes artifacts, other ranks
synchronize; clean up the process group on success and error.
Follow shared pinned-memory/worker/precision policy.

Test all four identical architectures/counts, branch-provenance rejection,
deterministic 16/8/8 sampling and fallback, static pools across epochs,
query-first inference, target boundaries and no validation/test gradients.
Test valid tail batches, checkpoint ties, exact lambda=1 identity,
lambda=0, stable ties, Top-N membership and out-of-N order, coverage ceiling,
hand-computed metrics, lock tampering/missing branches, and artifact reload.

Run compileall and the full pytest suite with mocked model downloads.
GPU equivalence and leakage tests must pass before real training.
Runtime evidence remains pending until these implementations exist.

## 9. Artifacts and expected commands

Per branch: outputs/e2b_pair_confidence/<representation>/<run_id>/seed_<seed>/.

~~~text
config_resolved.yaml, environment.json, metadata.json
inputs/{representation_freeze,embedding_manifest,split_manifest}.json
pairs/{train_static_top100,validation_top100}.pt
pairs/sampling_manifest.json
checkpoints/best.pt
training/{history,selection_trace}.json
selection.json
scores/{validation,test}_top100.pt
rankings/{validation,test}_{cosine,pairwise,fusion}.pt
metrics/{validation,test}.json
metrics/validation_n_lambda_grid.json
metrics/test_n_diagnostics.json
reliability/{validation,test}.json
failure_cases/
~~~

Store score IDs, cosine, raw logits, confidence, selected fusion score,
checkpoint/representation hashes, lambda, N and schema. Targets belong in
an analysis view and never the scorer input.

The joint export belongs under
outputs/e2b_pair_confidence/<run_id>/export/seed_<seed>/ and contains:

- cosine_results.csv, pairwise_results.csv, fusion_results.csv; each exactly
  four M1–M4 rows, with columns representation,R@1,R@2,R@4,R@8 in percent.
- metadata.json mapping CSV rows to integer hits, n_queries, split, seed,
  N/lambda, protocol and checkpoint hashes; validation and test stay separate.
- selected M3/M4 checkpoints and all four selected pair-head checkpoints.
- resolved configs, seeds, split manifest, global lock, training logs,
  source/environment provenance, scores, rankings and analysis files.
- raw embeddings or portable relative-path manifests with their required
  tensors included; no broken absolute paths to another machine.

A top-level multi-seed summary is separate from the required four-row
per-seed CSVs. Verify the export from a fresh directory by reloading
artifacts, checking hashes, and recomputing all CSV metrics. Do not fill
missing results with zero.

Expected interfaces after implementation:

~~~bash
torchrun --standalone --nproc_per_node=2 scripts/run_e2b.py --config configs/cub_e2b_m1.yaml --stage validation
torchrun --standalone --nproc_per_node=2 scripts/run_e2b.py --config configs/cub_e2b_m2.yaml --stage validation
torchrun --standalone --nproc_per_node=2 scripts/run_e2b.py --config configs/cub_e2b_m3.yaml --stage validation
torchrun --standalone --nproc_per_node=2 scripts/run_e2b.py --config configs/cub_e2b_m4.yaml --stage validation
python scripts/lock_e2ab.py --run-manifest outputs/e2b_pair_confidence/<run_id>/run_manifest.json
torchrun --standalone --nproc_per_node=2 scripts/run_e2ab_final.py --lock outputs/e2b_pair_confidence/<run_id>/global_lock.json
python scripts/verify_e2ab_outputs.py --export outputs/e2b_pair_confidence/<run_id>/export
~~~

The branch configs/run manifest declare seeds and execute the corresponding
roster before locking. These scripts are planned, not currently implemented.
Completion requires every primary result, checkpoint, log and reproducibility
artifact, irrespective of whether fusion improves Recall.
