# E2A-M3: CLS + mean patch projection plan

Protocol: e2ab-v2. Status: proposed implementation; training is pending.

## 1. Representation

Follow [the shared protocol](e2a_plan.md) and
[the new specification](../experiments/e2a-e2b-new.md).
Join final CLS c_i and mean patch p_i by image ID:

~~~text
p_i = mean(final_patch_tokens_i, patch_dimension)   # FP32, [768]
v_i = W concat(c_i, p_i) + b
z_i = v_i / ||v_i||_2
W: [768,1536]; b: [768]
~~~

Use Linear(1536,768,bias=True), Xavier-uniform weights and zero bias.
Concatenation order is CLS, mean_patch. There is no separate normalization
of the two inputs, hidden layer, activation, or dropout. Normalize the output
for the metric loss and retrieval. Backbone and cached input tensors are frozen.

This is the project's supervised DINOv3 fusion adaptation. It combines a new
representation construction with a published metric-learning objective;
the full system is not a faithful reproduction of that paper.

## 2. Proxy Anchor objective shared with M4

Use [Proxy Anchor v1, Eq. (4)](https://arxiv.org/pdf/2003.13911v1) with
100 learnable class proxies and L2-normalized features/proxies.
Let P be all proxies, P+ the proxies whose classes occur in the global batch,
and X_p+ / X_p- its matching/nonmatching embeddings:

~~~text
L_positive = (1 / |P+|) sum_{p in P+}
             log(1 + sum_{x in X_p+} exp(-alpha * (cos(x,p) - delta)))
L_negative = (1 / |P|) sum_{p in P}
             log(1 + sum_{x in X_p-} exp( alpha * (cos(x,p) + delta)))
L = L_positive + L_negative
alpha = 32; delta = 0.1
~~~

A proxy without positives is excluded only from the positive denominator.
Use a masked log-sum-exp including the constant-zero logit for numerical
stability; an empty term is zero. Empty global batches are invalid.
Initialize proxies with Kaiming normal, mode fan_out, as in the inspected
author implementation. See [sources](sources.md) for pinned commit/license.

Keep proxies in the training module and optimizer on the assigned GPU.
They are not retrieval embeddings and are discarded at inference.
Report 1,180,416 deployed projection parameters + 76,800 training-only proxies
= 1,257,216 optimized parameters. Counts are descriptive, not winner rules.

## 3. Controlled training recipe

M3/M4 use these same project defaults, selected before the new experiment.
They adapt the original training setup; they are not claimed as paper defaults.

| Setting | Fixed value |
|---|---|
| Input partition | Development/Train only, 4,687 images |
| Global batch | 20 distinct classes × 4 distinct images per class = 80 |
| Two-T4 local batch | 40 images per rank |
| Steps per epoch | ceil(4687 / 80) = 59 |
| Epochs | 30 |
| Optimizer | AdamW, lr 1e-4, betas (0.9,0.999), eps 1e-8 |
| Weight decay | 1e-4 on representation weights; zero on biases/proxies |
| Schedule | Cosine to zero, stepped after each optimizer update |
| Precision | FP32 representation, normalization, proxies, loss, backward |
| Gradient clipping | Global parameter-gradient norm 5.0 |
| Validation | Exact cosine retrieval after each epoch |
| Split seed | 42, identical for all runs |
| Training seeds | Primary 42; publication roster [42,43,44] per shared plan |

For each global batch, sample 20 classes uniformly without replacement and
four images per selected class without replacement. Images may recur across
batches. Every epoch has the declared number of sampled batches rather than
a guaranteed full pass over all images. Persist seed, epoch, batch indices,
sampler version and repetition counts. M3/M4 use the same sampled image
sequence for a given seed. No augmentation is added to cached inputs.

Use DDP with an autograd-aware global embedding gather and global labels.
The nonlinear loss must see all 80 examples. Synchronize both representation
and proxy gradients; compare loss and every gradient with a one-device
reference before accepting the implementation. Do not assume an arbitrary
world-size multiplier is correct. Proxies must be registered inside the
DDP-managed training module.

Abort on invalid labels, nonfinite loss/gradients, or corrupted inputs.
A development-only resource adjustment must preserve global batch/objective
semantics and be applied to both M3/M4 before comparison.

## 4. Checkpoint selection and freeze

Fit only on Train. Validation consists of the shared 1,177 image IDs from
classes 0–99, with its own query/gallery and no training images.
Select maximum integer validation Hits@1, then Hits@2, Hits@4, Hits@8,
then earliest epoch. Log all 30 epochs and the complete decision trace.

The primary architecture/loss/optimizer search has one fixed setting.
If additional development tuning is undertaken, declare equal candidate
budgets for M3/M4, log every trial, and update the run manifest before Final
Test. Never tune from test performance or pick a favorable seed.

After selection freeze the projection, export Train/Validation [N,768]
embeddings, and pass them to E2B-M3. Do not refit on Train+Validation.
For each declared training seed use the matching M3 checkpoint in E2B.
The shared global lock must include all four E2B branches before M3 test
embeddings or test rankings can be generated.

## 5. Implementation and artifact contract

Implement strict config in configs/cub_e2a_m3.yaml, CLSMeanPatchProjection in
models/representations.py, ProxyAnchorLoss in models/metric_learning.py,
and the generic training/representation.py loop. Add data/fused_cache.py
with identity-based CLS/mean joins and complete provenance checks.
Extend scripts/{train_e2a,evaluate_e2a,run_e2a}.py rather than duplicate
evaluation logic.

Output root: outputs/e2a_fusion/<run_id>/seed_<seed>/.

~~~text
config_resolved.yaml
environment.json
metadata.json
inputs/{cls_manifest,mean_patch_manifest,split_manifest}.json
checkpoints/best.pt
training/{history,sampling_manifest,selection_trace}.json
embeddings/{train,validation}.pt
embeddings/manifest.json
rankings/validation_top100.pt
metrics/validation.json
freeze.json
~~~

Checkpoint: projection/proxies, optimizer/scheduler, epoch/global step, RNG and
sampler states, selected validation hits, cache/split/config/code hashes,
parameter counts and source provenance. Write atomically on rank 0.
Embedding manifest: raw/normalized convention, input concatenation order,
upstream hashes, checkpoint hash, IDs, shape, dtype and partition.
Final evaluation later appends test embeddings, rankings and metrics.

## 6. Tests and acceptance

Test shuffled-row cache joins, missing/duplicate IDs and provenance drift;
hand-computed projection/input order; initialization, dimensions, counts,
and gradients only to projection/proxies. Test Proxy Anchor with missing
positive classes, empty negative sets, large logits, and finite gradients,
against Eq. (4) and the independently inspected implementation.

Test class-balanced sampling, all RNG restoration on resume, validation
epoch ties, and failure when Final Test or Validation enters the optimizer.
Run two-rank CPU/mock gradient tests where possible and CUDA equivalence
tests on both T4s. Confirm proxy parameters are synchronized and saved.

Verify saved rankings reproduce Hits@K, checkpoint/config/cache hashes match,
and the final stage rejects a lock missing any E2B branch. Run compileall,
the complete pytest suite and mocked integration smoke before real training.

Expected command after implementation:

~~~bash
torchrun --standalone --nproc_per_node=2 scripts/run_e2a.py --config configs/cub_e2a_m3.yaml --stage validation
~~~

Accept M3 for E2B when its selected checkpoint, development embeddings,
selection trace and all integrity evidence exist. An improvement over M1/M2
is not an acceptance condition.
