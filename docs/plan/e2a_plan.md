# E2A–E2B shared protocol and representation ablation plan

Protocol version: e2ab-v2, revised 2026-09-20. Status: implementation plan;
experiments and runtime acceptance are pending.

## 1. Authority, scope, and questions

Follow [the new experiment specification](../experiments/e2a-e2b-new.md),
[README](../../README.md), and [AGENTS](../../AGENTS.md).
This plan replaces the previous selection workflow. Earlier versions remain
in Git history; older experiment notes cannot override the new specification.
The active pipeline consists of E2A and E2B only.

E2A asks how four frozen-backbone representation pipelines compare under
cosine retrieval. E2B asks how those representations affect learned pair
compatibility and the change from cosine to fusion. Carry all M1–M4 into E2B.
Final performance is an outcome to report, never a rule for selecting the
representation used by another branch.

| Method | Construction | Output dimension | Learned representation parameters |
|---|---|---:|---:|
| [M1](e2a_m1_plan.md) | Final CLS | 768 | 0 |
| [M2](e2a_m2_plan.md) | Mean final patch | 768 | 0 |
| [M3](e2a_m3_plan.md) | CLS + mean patch, affine projection | 768 | 1,180,416 |
| [M4](e2a_m4_plan.md) | Learned attention over final patches | 768 | 197,121 |

M3/M4 use supervised labels and different parameter budgets from M1/M2.
Consequently this is a representation-pipeline ablation, not a causal estimate
of pooling alone. Report training-only proxy counts separately.

## 2. Exact data partition

Read original CUB manifests; convert class IDs 1–200 to 0–199 exactly once.
Expected totals from the local manifest: 11,788 images, 5,864 development
images (classes 0–99), and 5,924 Final Test images (classes 100–199).
Do not use the official image-level train_test_split.txt for retrieval splits.

For each development class c with n_c images:

1. Sort its image IDs ascending.
2. Set n_train = floor(0.8 * n_c + 0.5); n_validation = n_c - n_train.
   Implement with integer arithmetic (8 * n_c + 5) // 10.
3. Permute using NumPy PCG64 initialized with SeedSequence([42, c]).
4. Assign the first n_train IDs to Train and the rest to Validation.
5. Persist sorted ID lists, per-class counts, RNG/version, and a SHA-256
   manifest hash. Every branch must consume this manifest, not regenerate it.

This specified rounding gives 4,687 Train and 1,177 Validation images on the
current manifest. Counts alone do not establish identity with older splits;
do not reuse a historical hash without comparing its actual IDs. The split
seed stays 42 for every training seed.

Train/Validation have the same classes and disjoint images. Validate full
coverage and zero overlap; each evaluation split is its own query and gallery.
Treat corrupt/missing images as a blocking data-integrity error; never silently
drop images, alter a denominator, or resample a more favorable split.

Final Test images, tokens, and learned embeddings are processed only after
the global E2A/E2B lock. Metadata-only checks of class boundaries and counts
are permitted beforehand. Training/tuning datasets must be filtered before
decoding and must never expose Final Test examples.

## 3. Backbone, preprocessing, and cache controls

Use frozen facebook/dinov3-vitb16-pretrain-lvd1689m at authenticated, verified
revision 5931719e67bbdb9737e363e781fb0c67687896bc. Its config specifies
12 layers, 12 attention heads, hidden dimension 768, MLP dimension 3072,
and four register tokens; the Hub inventory and meta-device structure check
agree on 85,660,416 parameters. See the complete
[model/processor verification record](dinov3_model_verification.md) for
source URLs, file SHA-256 values, license and limits of the checks.

Lock the verified preprocessing order: explicit Pillow RGB conversion ->
CHW tensor -> FP32 rescale 1/255 -> direct 224×224 bilinear resize with
antialias=True -> normalize with mean [0.485,0.456,0.406] and
std [0.229,0.224,0.225]. The checkpoint's center-crop and RGB-conversion
flags are null; crop is inactive, while the dataset loader must explicitly
ensure RGB. The processor already specifies 224×224, so no resolution
override is needed. Preserve one deterministic whole-image view across all
branches, with no bounding-box/part crop or random image augmentation.
Record the exact decoder/processor backend, versions, resize order and
precision; rescaling after a uint8 resize is not the verified pipeline.

The model adapter must return the same final normalized transformer output
for every method: CLS [B,768], register [B,4,768], patch [B,196,768] at
224×224. Sequence order is CLS, register, patch (201 tokens total).
Use outputs.last_hidden_state after final model.norm: index 0 is CLS,
indices 1:5 are registers, and indices 5: are patches. pooler_output is
the CLS slice. Do not substitute hidden_states[-1], which is pre-final-norm
in the inspected backend, or apply the final LayerNorm twice.
Transformer output normalization is distinct from retrieval L2 normalization.
Keep the backbone in eval mode as well as disabling gradients; configured
position rescaling can otherwise activate in training mode.

Before extraction, hash-check the downloaded model.safetensors against the
Hub-reported SHA-256 in the verification record. The audit checked metadata,
synthetic CPU preprocessing and meta-device shapes, not pretrained numerical
outputs. Recheck these contracts on the Python 3.11/two-T4 runtime and pin
its compatible software versions; the local audit environment is not a
validated target-runtime lock.

Extract development CLS and full patch tokens together on two GPUs.
Derive the M2 mean and M3 patch input from exactly those cached patches.
Store sharded raw CLS and patch tensors in FP32 with image-ID indexing;
this avoids branch-specific cache quantization. Development patch storage is
3,530,784,768 bytes before metadata. Memory-map shards and stage batches.
The public method outputs need only [N,768] embeddings.

Use common FP16 AMP backbone inference when a development-only numerical
comparison against FP32 passes: on a fixed 128-image Train subset, require
finite outputs, per-image CLS/patch relative L2 error <= 1e-2, and CLS and
mean-patch embedding cosine agreement >= 0.9999. These are project
engineering tolerances, not literature claims. If any fail, use FP32
extraction for all methods and regenerate the common cache before training.
Record results and tolerances; do not infer fidelity from an FP16-to-FP32 cast.

Cache provenance includes tensor hashes, image IDs, split hash, processor,
revision, source-layer semantics, dtype, and schema e2ab-v2.
All joins are by image ID. Legacy caches require full equivalence checks and
development-only exports; historical metrics and selection artifacts are
not acceptance targets.

## 4. Cosine retrieval and denominators

For finite nonzero raw v, use FP32 z = v / ||v||_2. Reject norms <= 1e-12.
The common scorer is s(q,g) = z_q dot z_g. Disable TF32 for the FP32 reference
evaluator. Mask identical image IDs before ranking; ties use ascending
gallery image ID. Use chunked exact GPU scoring without approximate search.

For K in {1,2,4,8}, h_q(K) is 1 if any non-self top-K candidate shares q's
class, otherwise 0. Save integer Hits@K = sum_q h_q(K) and n_queries.
Recall@K = Hits@K / n_queries; CSV percentages are 100 * Recall@K.
Validation uses 1,177 queries; Final Test uses 5,924. Reject one-hit
disagreements even when printed percentages round to the same value.

Save top min(100, gallery_size - 1) candidate IDs and scores, query IDs, and
per-query hits. Fixtures must have a valid positive for every query;
production gallery coverage is mandatory. Recompute every metric from saved
rankings. Identical-cache reruns on a pinned runtime must yield identical
rankings; cross-hardware numeric differences are measured and disclosed.

## 5. Training, validation selection, and freeze order

M1/M2 require no optimizer or checkpoint selection. M3/M4 use the common
Proxy Anchor recipe in [M3](e2a_m3_plan.md), with exactly the same sampler,
training budget, precision, and epoch-selection rule.
Select M3/M4 checkpoints by maximum validation integer Hits@1, then
Hits@2, Hits@4, Hits@8, then earliest epoch. No test metric participates.

Freeze every representation before its E2B branch trains. Export all four
Train/Validation embedding manifests. Never refit either representation or
pair head on Train+Validation after selection; validation independence must
be preserved through the entire pipeline.

The primary run uses training seed 42. For publication robustness, plan
seeds [42,43,44] with the same fixed split and settings, recording all runs.
Each learned representation is retrained for each seed; its E2B head uses the
matching upstream seed. M1/M2 features are reused; their pair heads still vary.
Do not choose a favorable seed. If resources restrict execution to seed 42,
declare that budget before Final Test and report single-seed limitations.
Additional seeds cannot be added in response to Final Test performance.

The global run manifest records the selected seed roster, all four
representation manifests, all four pair-head selections per seed, checkpoint
hashes, lambda values, preprocessing, sampler, evaluator, code/config hashes,
and planned analyses. Missing branches or mismatching hashes prevent final
evaluation. Lock all planned seeds before any Final Test scores are read.

Execution dependency:

Development manifest -> common tokens -> E2A M1–M4 validation/freeze ->
E2B M1–M4 validation/freeze -> global lock -> Final Test -> export/audit.

## 6. Implementation work and expected commands

At this revision the repository has no src/, scripts/, configs/, or tests/
implementation. All entry points below are proposed interfaces, not executable
claims. Implement only functional modules; mock external checkpoints in tests.

1. Add strict E2A config, CUB split/manifest, token adapter, shared cache,
   GPU exact retrieval, and artifact validation.
2. Implement M1/M2 and validate their deterministic development outputs.
3. Implement shared Proxy Anchor/DDP training, then M3/M4.
4. Add per-representation E2B input exports and freeze records.
5. Implement [E2B](e2b_plan.md), global lock validation, and a joint final
   evaluator that produces all 12 primary rows per seed.
6. Implement a report/export auditor checking every required artifact.

Expected interfaces after implementation:

~~~bash
python -m compileall src
python -m pytest
torchrun --standalone --nproc_per_node=2 scripts/run_e2a.py --config configs/cub_e2a_m1.yaml --stage validation
torchrun --standalone --nproc_per_node=2 scripts/run_e2a.py --config configs/cub_e2a_m2.yaml --stage validation
torchrun --standalone --nproc_per_node=2 scripts/run_e2a.py --config configs/cub_e2a_m3.yaml --stage validation
torchrun --standalone --nproc_per_node=2 scripts/run_e2a.py --config configs/cub_e2a_m4.yaml --stage validation
~~~

Use a shared scripts/run_e2ab_final.py --lock <global_lock.json> after E2B.
All component --stage test modes must require that lock; no E2A-only test
stage can bypass completion of E2B. No runner spawns a second torchrun inside
an existing distributed job.

## 7. Device policy and acceptance

Use one process per T4 for extraction, training, query scoring, and suitable
metric workloads. Keep batches and intermediate tensors on the assigned GPU.
DataLoader workers perform CPU decode/indexing only; use pinned memory,
nonblocking transfers, persistent workers and prefetch_factor when workers > 0.
Use batch-size/chunk tuning based on Train memory/throughput only.

For inference use non-padding sharding and merge by image ID; every real
query appears once. For training ensure matching step counts and record any
tail padding. Global metrics reduce sums/counts, not unweighted rank means.
Proxy Anchor requires differentiable global-batch semantics; BCE does not.
Use FP32 head/loss computation for the initial controlled recipe.

Record cuDNN benchmark/deterministic flags, RNG states, dependency versions,
hardware and throughput. Enable benchmarking for fixed-shape performance
runs; a deterministic verification mode may disable it. Do not promise
bitwise equality across hardware, kernels, or precision modes.

Run compileall and the complete pytest suite once implementation exists.
Required tests cover split boundaries/rounding, no leakage, token boundaries,
zero norms/nonfinite inputs, labels/ID joins, frozen backbones, gradient
equivalence across two ranks, checkpoint selection, global-lock rejection,
rank ties and metric/artifact recomputation. GPU tests require the target
runtime; report skips explicitly rather than counting them as passes.
Add processor regressions on nonsquare RGB and explicitly converted L/RGBA
inputs, rescale-before-resize/antialias behavior, final-LayerNorm token
selection, eval-mode backbone enforcement, and downloaded weight-hash checks.

## 8. Evidence and publication limits

Save each method below outputs/e2a_cls/, e2a_mean_patch/, e2a_fusion/, or
e2a_attention_pool/ using <run_id>/seed_<seed> directories. Include resolved
config, environment, shared split reference, cache manifest, metadata,
validation metrics/rankings, and freeze.json; M3/M4 also include best.pt and
training history. Add test artifacts only inside the locked final campaign.

The new protocol prevents further test-driven selection. It cannot erase
earlier exposure: previous repository plans state that classes 100–199 were
inspected and used to choose a representation. Preserve that disclosure,
verify available run history, and describe these as unseen *training classes*.
Do not claim an historically untouched holdout without evidence.

Within-class-set validation, unknown overlap with backbone pretraining, one
dataset/backbone, supervised-versus-fixed representation budgets, and
representation-dependent candidate sets limit generalization.
See [the verification and reporting plan](research_verification.md) for
statistical analyses, source checks, and outstanding evidence.
