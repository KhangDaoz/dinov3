# Research verification and publication evidence plan

Audit date: 2026-09-20. Protocol: e2ab-v2.
Scope: all six implementation plans and the authenticated model audit in
this directory.

## 1. Interpretation of this audit

The new plans are checked against
[the experiment specification](../experiments/e2a-e2b-new.md),
[README](../../README.md), and [AGENTS](../../AGENTS.md).
[Primary sources](sources.md) document the literature and the
NeurIPS/CVPR reporting references consulted.

There is no universal checklist that certifies a paper as internationally
acceptable. A consistent plan establishes what should be tested; empirical
claims require actual runs, artifacts and reviewer scrutiny.

This revision has no implemented runners/configs/tests, trained checkpoints,
or new Recall results. Plan checks, source verification, runtime correctness
and scientific conclusions therefore have separate statuses.

## 2. Protocol audit and evidence gates

| Check | Plan-level resolution | Runtime or reporting evidence still needed |
|---|---|---|
| Scope | E2A plus four E2B branches | All 12 primary rows per seed |
| Development partition | Per-class rounded 80/20, shared split seed 42 | Saved IDs/hash, full disjointness tests |
| Final Test access | Global lock after every declared branch/seed | Access log and lock verification tests |
| E2A handoff | M1–M4 all frozen and exported | Four complete representation manifests |
| M3/M4 optimization | Common loss/batch/selection recipe | Loss/gradient oracle and DDP tests |
| Pair construction | Train-only 16/8/8 static mining | Pair IDs, targets, pool/repetition statistics |
| E2B controls | Same 768-D embeddings, 2304-input MLP and tuning budget | Parameter/config equality, seed roster |
| Metrics | Exact cosine, ID self-exclusion, R@1/2/4/8 | Recomputed integer hits and denominators |
| Candidate scope | Top-100 pair/fusion reranking | Coverage, pair overlap, N diagnostics |
| Selection | Validation checkpoint/lambda only | Full decision traces and no test tuning |
| Deliverables | Three CSVs, six learned checkpoints per seed | Portable bundle reloaded and verified |
| Resources | Two T4s, explicit precision and global batches | Peak memory, timing, GPU-hours, skipped tests |
| Attribution | Sources/license bytes and authenticated DINOv3 metadata verified | Local weight digest and target-runtime validation |
| Claims | Pipeline effects, calibrated-score claim withheld | All positive/null/negative results disclosed |

No empirical gate is marked passed merely because it appears in a plan.

## 3. Statistical analysis declared before Final Test

Primary estimands are the four fusion-minus-cosine Recall@1 differences,
one per representation. Each uses exactly the same queries/gallery within
that comparison. Report percentage-point differences and all constituent
Recall values, not only the largest gain. Pairwise-only and E2A
representation comparisons are secondary/descriptive.

The primary seed is 42. The publication roster is [42,43,44] on one fixed
split, with all learned upstream and pair modules rerun per seed.
Report every seed and arithmetic mean ± sample standard deviation (ddof=1)
for learned pipelines. Also report paired within-seed fusion-minus-cosine
deltas. Three seeds give a limited view of optimization variance.
M1/M2 cosine features are deterministic controls; duplicate identical rows
do not create independent training replicates.

If only seed 42 is funded, declare that choice before the global lock,
report it as a single-seed experiment, and omit fabricated standard
deviations. Do not stop or add seeds based on Final Test outcomes.

For seed-42 per-query hit vectors, compute paired query bootstrap intervals:
2,000 replicates, seed 42, sample n query indices with replacement, use
identical draws for all methods, and compute each replicate's mean hit
difference. Use percentile 2.5/97.5 bounds for exploratory 95% intervals.
Keep the gallery and trained models fixed. State that these intervals
describe query resampling conditional on that gallery; they do not measure
training, split, or gallery uncertainty.

Queries and candidates share images, so IID assumptions are imperfect.
Add a class-cluster sensitivity analysis: sample 100 query classes with
replacement, retain all queries of each sampled class with multiplicity,
and divide total sampled hits by total sampled query count. Use the same
class draws across methods and 2,000 replicates. This still holds gallery
fixed and does not establish generalization to every future bird dataset.

The four primary contrasts form one comparison family. Do not declare
statistical significance from four unadjusted 95% intervals. If reporting
familywise intervals, preregister Bonferroni percentile bounds
0.625/99.375 (98.75% per contrast) and use 10,000 paired bootstrap draws for
adequate tail resolution. Treat coverage as approximate under the stated
dependence assumptions; report effect sizes and seed variability prominently.
Do not claim a universally best representation from a post-hoc max over
methods, seeds, K values and N budgets.

These statistical procedures are project analysis choices. They do not
repair historical test exposure or establish probabilistic calibration.

## 4. Essential limits to include in the paper

- Earlier plan versions state that classes 100–199 were inspected and used
  to select a representation. The new run prevents additional test-driven
  selection, but cannot restore historical independence. Verify available
  logs and disclose that history; unseen training classes do not imply an
  untouched benchmark.
- Validation uses new images of training classes. It may favor settings
  that generalize differently to the unseen test classes. Keep the required
  split unchanged and state this limit.
- M3/M4 use supervised learning and unequal head capacities; M1/M2 do not.
  Conclusions apply to these pipelines, not exclusively to token semantics.
- Each representation defines its hard-negative and retrieval candidate
  pools. Identical mining rules do not imply identical pair observations.
  Coverage and pool overlap must accompany representation comparisons.
- Balanced mined pairs differ from natural retrieval pairs. A sigmoid
  confidence score does not automatically estimate a calibrated match
  probability. Reliability measurements must use the specified populations.
- One dataset, one backbone and one resolution do not support broad
  state-of-the-art or cross-domain claims. Comparisons to published numbers
  with different pretraining, training budgets or preprocessing are context,
  not controlled improvement estimates.
- The [CUB maintainers](https://www.vision.caltech.edu/datasets/cub_200_2011/)
  identify overlap risk with common pretraining sources. DINOv3 pretraining
  overlap is not verified here.

## 5. Paper and supplement contents

The paper should state the two research questions, precise equations,
experiment matrix, class/image split, sample counts, pretrained checkpoint,
frozen versus learned parameters, selection rules and candidate constraint.
Present three M1–M4 result tables, the per-branch fusion deltas, uncertainty
definitions, compute summary and limitations.

The supplement/export should include resolved configs, full epoch/trial
histories, selected epochs/lambdas, seed results, source/deviation ledger,
split IDs/hashes, executable environment and commands, metric definitions,
score/ranking hashes, deterministic qualitative categories and failure cases.
Report unsuccessful trials and total development cost as well as final runs.

Check venue-specific anonymity, format, references, submission policies and
asset terms at submission time. Dataset/model redistribution must follow the
actual terms. Record relevant tool assistance if required by the venue.
This audit does not evaluate novelty, prove theoretical results or guarantee
publication readiness.

## 6. Verification performed during this documentation revision

- Read and replaced all six old plans, removing operational test-based
  winner selection, M1-only handoff and evidential controls.
- Inspected the local CUB class-label manifest and verified total and
  rounded per-class partition counts; no test images or retrieval results
  were used for this check.
- Checked Proxy Anchor Eq. (4) and the author loss implementation/defaults;
  resolved its commit and MIT license, and resolved the IDML source commit.
- Read the public DINOv3 model card, supplied related papers, CUB source,
  NeurIPS reporting checklist and CVPR author guidelines.
- Resolved the earlier HTTP 401 using user-authorized access; verified the
  exact DINOv3 revision and downloaded config, processor, model-card and
  license bytes. Recorded the Hub weight digest as remote metadata, not a
  locally verified weight hash.
- Ran synthetic CPU processor/reference comparisons and meta-device model
  shape/parameter checks; see [model verification](dinov3_model_verification.md).
  No pretrained weights or CUB test images were used.
- Ran local documentation checks for whitespace, relative links, code
  fences, obsolete positive selection instructions, and consistency of
  parameter, sample, storage, pair and batch arithmetic.

The initial plan revision passed checks for 8 plan/reference documents and
38 relative links (including README/AGENTS references). After the authenticated
model audit, the checks passed for 9 documents and 42 relative links, with no
Hugging Face token pattern in documents or tracked changes. All four downloaded
metadata SHA-256 values and Git blob IDs matched the authenticated Hub records.
Verified arithmetic:
5,864 development = 4,687 Train + 1,177 Validation, and 5,924 Final Test;
M3/M4/pair-head parameter counts, 149,984 training pairs, 117,700 validation
pairs, the 2,528-pair tail batch, and shared patch-cache byte count.
The final whitespace check is git diff --check.

The current repository has no implementation suite to run, so compileall,
pytest, GPU parity, actual seed reproducibility, calibration, Recall values
and fresh-machine export verification remain pending. The checks above
validate documentation/source alignment plus synthetic processor and
meta-device structure behavior, not learned retrieval performance.

## 7. Future automated acceptance report

When implemented, scripts/verify_e2ab_outputs.py should emit a machine-readable
report with protocol/run hash, check ID, pass/fail/unavailable status,
artifact path/hash, observed/expected values and reasons. It must check all
branches/seeds, CSV schemas/percent units, exact hit recomputation,
split/pair boundaries, checkpoint selections, locked lambdas, provenance,
source/license records, and portable file dependencies.

Require nonzero exit on any missing required artifact or mismatched metric.
Keep a separate runtime status for GPU tests and a separate human review
status for claims, limitations, historical exposure and publication policy.
An automated green report never substitutes for those scientific judgments.
