# E2B Pair-wise Confidence Learning Implementation Plan

## 1. Objective and status

Learn a query--candidate compatibility score that addresses the limitations
of image-level evidential uncertainty observed in E1. Evaluate whether it
improves retrieval ranking and predicts retrieval correctness; these are
separate research questions, neither is assumed to succeed.

This plan defines the implemented E2B pipeline and the artifacts required for
a complete rerun from validation through benchmark evaluation.

## 2. Accepted inputs and inherited limitations

Use **E2A-M1 final CLS**, not M3: the accepted test-based selection artifact
`outputs/e2a_selection/test_selection.json` selects M1 with Hits@1
5,214/5,924 (Recall@1 88.0149%). M3 has 5,171 Hits@1. Verify selection,
config, manifest, cache, and ranking hashes before accepting inputs.

Reuse the pinned frozen DINOv3 ViT-B/16 checkpoint, revision
`5931719e67bbdb9737e363e781fb0c67687896bc`, processor at 224x224,
and the M1 CLS cache `[11788,768]`. L2-normalize features for pair inputs
and cosine retrieval; do not train or re-extract the backbone unnecessarily.
Register/patch tokens are not needed by the pair network.

Classes 100--199 were already used to select M1 and have been inspected and
rerun. They remain the fixed **evaluation benchmark**, not an untouched
final test. All E2B reports must disclose inherited representation-selection
bias. Do not additionally select checkpoint, lambda, seed, architecture,
sampling, or calibration from this benchmark.

## 3. Fixed data protocol and leakage barriers

- Development classes 0--99: 5,864 images.
- Canonical seed-42 stratified image-level split: 4,687 fit and 1,177
  validation IDs, matching accepted M1 IDs and split hash
  `d9455ac8948ff640f94017339681a51744f6351f1db053732b3153a4c9c958ee`.
- Benchmark classes 100--199: 5,924 images; ignore official CUB image split.
- Training pair endpoints must both belong to fit; validation pair endpoints
  must both belong to validation. Reject mixed fit/validation endpoints,
  benchmark endpoints, and self-pairs in all training/tuning loaders.
- Mine fit candidates using only fit query/gallery features. Existing E2A
  validation rankings cannot be used to train the pair head.
- Query/gallery are identical within each evaluated split, excluding self
  by image ID. Preserve exact cosine, stable gallery-index ties and R@1/2/4/8.
- Fit and validation contain the same classes but different images. This
  limits assessment of class-generalization to unseen classes; report it.
- Benchmark labels may be read by metric/failure-analysis code only after
  checkpoint and lambda selection, never by sampling or scoring code.

Keep the selected fit-trained checkpoint for benchmark evaluation; do not
automatically refit on all development images in this first E2B run.

## 4. Pair sampling and labels

Positive: distinct images of the same original class (`target=1`).
Negative: images from different classes (`target=0`). No absolute class ID,
image ID, rank index, uncertainty, or label is a network input.

Initial deterministic recipe per fit query and epoch:

- 16 positives, sampled uniformly from its other same-class fit images;
- 8 hard negatives, sampled from different-class candidates in its fit
  cosine Top-100;
- 8 random negatives, sampled from different-class fit images.

Use replacement when a pool is too small, record repeated-pair frequency,
and fall back to the random negative pool when no hard negative exists.
Fail if no valid positive/negative pool exists. This yields 149,984 ordered
pairs per epoch with a 1:1 positive/negative ratio. Persist sampling seed,
epoch, pair IDs/targets or a losslessly reproducible sampling manifest.

**Static hard-negative mining:** mine fit Top-100 once from frozen M1 cosine
features and persist/hash that candidate pool. Epochs resample negatives from
this same pool; never update it online using Pair Confidence or learned
fusion scores.

**Preregistered orientation:** inference always uses
`C(q,x) = sigmoid(MLP(concat(q,x,abs(q-x))))`, where MLP emits a logit.
The first feature is always the query and the second the candidate.
Random swap with seeded 50% probability is training augmentation only;
it does not guarantee symmetry. Measure swap disagreement diagnostically,
but do not switch inference to bidirectional averaging after inspecting the
benchmark. Bidirectional averaging requires a separate preregistered ablation.

Validation uses **all** 117,700 ordered validation Top-100 pairs at their
natural class-match prevalence, not a balanced resample. Persist pair IDs,
targets, cosine scores and prevalence. Use separate names for balanced
training metrics and natural-candidate validation metrics.

## 5. Pair Confidence Network and optimization

For normalized frozen features `a,b`:

```text
input = concat(a, b, abs(a-b))             # 2304 dimensions
Linear(2304,512) -> ReLU -> Dropout(0.1)
Linear(512,128)  -> ReLU -> Dropout(0.1)
Linear(128,1)                            # one logit
pair_confidence = sigmoid(logit)          # inference only
```

Use `BCEWithLogitsLoss` without class weights for the balanced training
recipe. Do not apply sigmoid before this loss. Initial recipe: seed 42,
30 epochs, AdamW LR `1e-4`, weight decay `1e-4`, cosine scheduler,
gradient clipping 5.0, global batch 4,096 pairs (2,048 per T4).
Record actual batch sizes and parameter count. Reject NaN/Inf gradients
with actionable epoch/batch diagnostics.

Use FP32 for the first pair-head training run to avoid introducing another
precision variable. AMP is a later separately declared optimization, not a
silent recipe change. Backbone remains frozen with no gradients.

Checkpoint rule: lowest natural-candidate validation BCE computed from
global summed FP64 loss/count; ties use the earliest epoch. Record validation
retrieval as secondary diagnostics, not an alternate epoch-selection rule.
Store model, optimizer, scheduler, epoch, metrics, sampling configuration,
feature/split/config hashes and Git/environment provenance atomically.

Because training pairs are balanced and selectively mined, the sigmoid is
not automatically a calibrated probability under natural retrieval
prevalence. Call it **pair confidence/compatibility score**, not proven
`P(retrieval correct)`. Evaluate calibration explicitly; temperature/prior
correction is outside the initial run and requires a separate plan.

## 6. Retrieval, fusion and validation selection

Create the baseline exact cosine Top-100 candidate list once per split.
Score only these candidates; pair inference receives features, not labels.
For each candidate apply the documented raw-score formula:

`score = lambda * cosine + (1-lambda) * pair_confidence`.

Do not add per-query min-max normalization or an unplanned affine transform.
Cosine lies in `[-1,1]`, confidence in `[0,1]`; lambda is a fusion coefficient,
not a calibrated mixture of probabilities. Persist the exact score convention.

Evaluate **N in {10, 20, 50, 100}** for Image Uncertainty, Pair-wise
Confidence, and Fusion. Candidates beyond N retain their baseline order and
cannot be promoted. N=100 remains the primary comparison, while all four
budgets are saved for controlled analysis. Stable score ties preserve original
cosine order. `lambda=1`
must return the original baseline ranking exactly without an extra sort.

After freezing the BCE-selected checkpoint, evaluate every N/lambda
combination on validation and test. Select lambda on the N=100 validation row from
`{0,0.25,0.5,0.75,0.9,1}` by integer Hits@1 -> Hits@2 -> Hits@4 -> Hits@8
-> larger lambda (prefer the smaller learned intervention when all Hits tie).
Save every N/lambda grid result and the decision trace. A winning lambda of 1 is a
valid null result; do not exclude it to force an improvement.

Save `selection.json` with checkpoint/feature/split/config hashes, selected
epoch, lambda, N, complete validation results and exact scoring/tie policy.
Benchmark runner verifies it before scoring. This is an **E2B tuning record**,
not a claim that the benchmark was untouched during E2A.

Initial ablations: cosine-only, Top-100 pair-only (`lambda=0`) and selected
fusion. Pair-only is candidate-constrained reranking, not exhaustive learned
retrieval over all query--gallery pairs. N=50 and alternate architectures or
sampling are deferred controlled ablations, not implicit additions.

## 7. Comparison methods and lineage

| Method | Representation/ranking | Role |
|---|---|---|
| B0 | Fixed M1 CLS + cosine | Shared baseline, 5,214 benchmark Hits@1 |
| U1 | Same B0, ascending candidate image uncertainty within Top-N | E1-style control |
| A1 | Raw 100-dimensional alpha vector + Euclidean L2 distance | Paper-faithful alpha/distance control |
| A2 | L2-normalized alpha vector + cosine | Secondary representation adaptation |
| P1 | Pair confidence reranking within cosine Top-100 | Component ablation |
| F1 | Validation-selected cosine + pair fusion within Top-100 | Primary E2B treatment |

Reuse E1 seed-42 full evidence, alpha and uncertainty, checking image IDs,
class ordering, shapes, dtype, actual resolved model revision and feature
provenance, not merely the historical `revision: main` config. Validation
must use outputs of an E1 tuning head trained on fit only; E1 final-head
outputs trained on all development images cannot tune validation controls.
Use corresponding final outputs for benchmark controls and disclose their
different training-data budgets relative to the fit-only pair network.

If cache provenance differs, stop reuse and document a controlled re-export
or retraining procedure before comparing. Historical E1 scores remain a
separate context table, not results of the new controlled B0 run. Do not
compare historical and current cache runs as if reranking were the only change.

Input inspection found the historical E1 cache differs from accepted M1 and
lacks full processor provenance. The implemented default is therefore
`controls.source: controlled_retrain`: prepare a new seed-42 E1-style linear
EDL tuning head on **raw accepted M1 CLS**, using the E1 30-epoch/lowest-EDL-
validation-loss recipe (LR 0.001, weight decay 0.0001, KL annealing 10 epochs,
FP32). Export full validation e/alpha/u, then retrain a freshly initialized
head on all development images for the selected epoch count **during the
validation stage**. Benchmark stage only forwards this frozen final head.
Artifacts remain under E2B `controls/edl/`, never overwrite historical E1.
This controlled rerun is not the accepted historical E1 result; report its
head training budget and precision explicitly. `reuse_e1` remains available
only when exact cache/provenance gates pass. Explicitly disabling controls
records them unavailable rather than filling the table with old scores.

Sources:

- [EDL, arXiv:1806.01768v3](https://arxiv.org/abs/1806.01768v3),
  evidence/alpha/uncertainty from the existing E1 implementation.
- [Evidential Transformers, arXiv:2409.01082v2](https://arxiv.org/html/2409.01082v2),
  CC BY 4.0, Sections 2.2/2.3. A1 follows Section 2.2: retrieve with raw alpha
  and ascending Euclidean L2 distance, without L2-normalizing alpha. It is
  paper-faithful in representation/distance only; reusing the E1 head on a
  frozen DINOv3 backbone is still a system-level adaptation, not a complete
  reproduction. A1 is an explicitly separate distance control: preserve
  class/image splits, self-match exclusion, stable ties and Recall@K, while
  keeping cosine unchanged for B0/P1/F1. A1 retrieves over its full gallery,
  not just B0's Top-100. Save its own Top-100 IDs and distances with a distinct
  distance schema, never mislabel them as cosine scores. A2 is the secondary
  normalized-alpha + cosine adaptation; do not select A1 versus A2 on the
  benchmark. Bhattacharyya experiments remain outside the initial scope.
- The pair MLP and fusion are the repository's specified experimental design,
  not claimed as a method from that evidential paper or as IDML reproduction.

No third-party source code is planned for copying. Record equations,
versions, licenses, deviations and implementation commit in provenance.

## 8. Metrics, reliability and failure analysis

Primary: benchmark F1 minus B0 Recall@1. Secondary: Recall@2/4/8, Top-100
candidate positive coverage, P1 and A1/U1 controls, per-query Hits and paired
bootstrap 95% intervals (2,000 samples, seed 42). Bootstrap is exploratory
under inherited selection bias; it does not remove that bias or measure
training variance. Chunk its computation for GPU memory safety.

Report reliability on two distinct populations:

1. All baseline Top-100 candidate pairs: AUROC, AUPRC, BCE, Brier score,
   fixed 10-bin ECE/reliability diagram and positive prevalence.
2. Baseline top-1 correctness: score that unchanged top-1 pair, AUROC/AUPRC
   and risk-coverage/AURC. Compare pair confidence with cosine and candidate
   image certainty `1-u` on the same queries. Post-fusion top-1 diagnostics
   are separate; never mix targets from one ranking with scores from another.

Save undefined metric reasons when a tiny fixture has only one target class;
do not manufacture a numeric AUROC. Do not report pooled candidate-pair
observations as independent query-level statistical samples.

Select successes/failures by deterministic categories and image-ID ordering:
wrong->correct, correct->wrong, still-wrong, still-correct. Persist IDs,
labels, image paths, baseline/fused candidate IDs, cosine, confidence and
fusion scores. Export PNG contact sheets showing query, cosine Top-1 and
reranked Top-1, plus SVG Recall@1 curves over N and lambda. Labels are for
analysis only. Do not cherry-pick after viewing
qualitative results. Report single-seed results, not seed mean +/- std.

## 9. Implementation phases and proposed files

1. **Input gates:** strict E2B config, accepted M1 cache/selection loader,
   canonical split checks, fit-only Top-100 mining and pair manifests.
2. **Pair head:** MLP, seeded sampler, BCE/DDP training, validation BCE,
   atomic checkpoint and numerical diagnostics.
3. **Validation:** pair score export, exact constrained reranking, lambda
   grid and tuning record. Accept artifacts before benchmark execution.
4. **Benchmark/report:** frozen scoring, controls, reliability, bootstrap,
   failure manifests, portable bundle and `reports/e2b.tex`.

Proposed additions (create only when functional):

```text
configs/cub_e2b.yaml
src/uncertainty_retrieval/config_e2b.py
src/uncertainty_retrieval/data/pair_cache.py
src/uncertainty_retrieval/sampling/{__init__,pairs}.py
src/uncertainty_retrieval/models/pair_confidence.py
src/uncertainty_retrieval/training/pair_confidence.py
src/uncertainty_retrieval/evaluation/pair_confidence.py
scripts/{prepare_e2b_pairs,train_e2b,evaluate_e2b,run_e2b}.py
scripts/prepare_e2b_controls.py
tests/unit/test_{config_e2b,pairs,pair_confidence,pair_reranking,e2b_artifacts}.py
tests/integration/test_e2b_smoke.py
reports/e2b.tex
```

Reuse CUB loading/split logic, cache hashes, environment metadata, cosine
ranking, AUROC/AUPRC/AURC and paired bootstrap where semantics match.
Do not route E2B through E1 config or Proxy Anchor training.

## 10. CUDA/DDP and correctness tests

Use both T4s for independent pair workloads. BCE is additive: shard global
pair batches evenly across DDP ranks; no differentiable global gather is
needed. Ensure equal step counts and sample-weighted global loss reductions.
For training tail padding, record duplicated samples or use an explicitly
equal-sized deterministic batch policy. Distributed validation/inference
must cover each real pair/query exactly once without padding duplicates.

Pair features/batches/loss remain on the local GPU; cached CPU features may
be staged once per process when memory permits. CPU workers never allocate
CUDA tensors. Use pinned memory, nonblocking transfers, persistent workers,
prefetching and configurable inference chunks. Split query rows across GPUs,
merge by IDs and original candidate position, write artifacts only on rank 0.
Always clean up the process group, including error paths.

Focused unit tests must cover:

- no fit/validation/test endpoint leakage or self-pairs; label correctness;
  deterministic mining, hard-negative fallback and orientation sampling;
- `[B,2304]` inputs, finite logits/BCE gradients, frozen features, sigmoid
  range and model checkpoint/config/cache/split mismatch rejection;
- lambda endpoints, exact B0 identity at lambda=1, stable ties, Top-N
  membership preservation and hand-computed Recall/Hits;
- static fit mining unchanged across epochs; fixed query-first inference
  despite training swaps; no automatic bidirectional averaging;
- A1 raw-alpha Euclidean distances on a hand-computed fixture, no alpha
  normalization, full-gallery ranking and distance-schema correctness;
- shard merge without duplicates/missing pairs, score/ID alignment;
- FP32 Recall round-off acceptance but rejection of one-hit mismatch;
- full e/alpha/u reuse validation and rejection of final-head validation
  tuning artifacts; benchmark labels unavailable to scorer/trainer;
- required outputs and portable export; do not declare success before
  writing/verifying every required artifact.

Add a separate two-rank BCE loss/gradient equivalence test against one-device
global-batch computation (disable dropout for equivalence). It is not part of
the initial tiny test subset. Tiny integration smoke uses mocked checkpoints,
one optimizer step and synthetic pairs; no real model download.

## 11. Artifact layout and execution contract

```text
outputs/e2b_pair_confidence/seed_42/
├── config_resolved.yaml, environment.json, metadata.json
├── inputs/{m1_manifest,e2a_selection,e1_provenance}.json
├── split/{fit_image_ids,validation_image_ids}.pt
├── pairs/{fit_sampling_manifest,validation}.pt
├── checkpoints/best.pt
├── selection.json
├── scores/{validation,test}_top100.pt
├── rankings/{validation,test}_{baseline,pair,fusion}.pt
├── rankings/{validation,test}_fusion_n{N}_lambda{value}.pt
├── metrics/{validation,test}_topn_lambda_grid.json
├── controls/{validation,test}_U1_n{N}.pt
├── failure_cases/{split}_{method}_n{N}.json
├── figures/{split}_topn_lambda_recall1.svg
├── figures/failure_cases/*.png
└── export/                   # portable evidence bundle, not just summary
```

Scores contain query/candidate IDs, cosine, raw logits, confidence, targets
for analysis, final scores, checkpoint hash, lambda, N and split. Rank
artifacts retain candidate IDs/scores and per-query Hits. Inputs include the
CLS cache path/hash; raw embeddings need not be duplicated. Export includes
configs, tuning record, checkpoints, scores, rankings, metrics and failures;
print absolute output paths and verify required files before ZIP handoff.

After implementation, expected commands are:

```bash
python scripts/run_e2b.py --config configs/<config_name>.yaml --stage validation
python scripts/run_e2b.py --config configs/<config_name>.yaml --stage test
```

Validation prepares static fit pairs, prepares controlled EDL comparison
heads when enabled, trains the pair head and selects checkpoint/lambda.
Test reuses the frozen tuning record and never trains. Export is written to
`outputs/e2b_pair_confidence/seed_42/export/` after required-file verification.
The entry points now exist; execute on the target runtime after input setup.

Acceptance: reproducible validation selection; no endpoint leakage; intact
score/ranking/provenance artifacts; common baseline reproduced from accepted
M1 rankings; complete reliability/negative-result reporting. Lambda=1,
negative Recall deltas, or uncalibrated confidence are valid findings, not
reasons to silently alter the recipe.
