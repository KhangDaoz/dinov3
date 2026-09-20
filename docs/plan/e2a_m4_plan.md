# E2A-M4: attention pooling representation plan

Protocol: e2ab-v2. Status: proposed implementation; training is pending.

## 1. Representation

Follow [the shared protocol](e2a_plan.md) and
[the new specification](../experiments/e2a-e2b-new.md).
For final patch matrix H_i with shape [196,768]:

~~~text
t_ij = w2^T tanh(W1 h_ij + b1) + b2
a_ij = softmax_j(t_ij)
v_i  = sum_j a_ij h_ij
z_i  = v_i / ||v_i||_2
~~~

Implement Linear(768,256), tanh, Linear(256,1), and weighted summation of
the original patches. Softmax is over patch positions only. Exclude CLS and
all four register tokens. There is no value/output projection, CLS fusion,
dropout, or multi-head attention. Initialize weights with Xavier uniform and
biases with zero using the run seed.

Deployed parameters: (768×256+256)+(256+1) = 197,121.
With 100 training-only 768-dimensional proxies, optimized parameters total
273,921. The scalar score bias cancels under softmax; retain it to preserve
the specified architecture and report its redundancy.

This single attention scorer is a project design, not the DINOv3 author's
attentive probe or an IDML/EDL method. See [sources](sources.md).

## 2. Common inputs and training

Use the shared development patch cache consumed by M2 and M3. Do not repeat
mean vectors to synthesize patches. Every shard must match the common
checkpoint, processor, source-layer semantics, extraction precision, CUB
manifest and split hash.

Expected development shape across shards is [5864,196,768].
The primary cache is FP32; numerical extraction checks and memory/storage
policy are defined once in the shared plan. Do not introduce M4-only FP16
storage. Load CPU shards with memory mapping and transfer only local batches.

Train the attention scorer and proxies using exactly the [M3 recipe](e2a_m3_plan.md):
Proxy Anchor alpha=32, delta=0.1; 20×4 global batches (40 per T4),
59 steps/epoch, 30 epochs, AdamW lr=1e-4, cosine schedule, clipping=5.0,
FP32 heads/loss/backward, matching sampler image IDs and seed roster.
Use the same weight-decay groups, proxy initialization and autograd-aware
global-gather loss. Backbone gradients remain disabled.

M3/M4 share labels, optimizer, sampling and checkpoint policy; their deployed
parameter counts differ. Report this when interpreting attention gains.
An additional parameter-matched architecture would be a separately planned
ablation, not a silent modification of M4.

## 3. Validation and freeze

Train on the 4,687 Train images, select on the shared 1,177 Validation images.
Rank checkpoints by integer validation Hits@1, Hits@2, Hits@4, Hits@8, then
earliest epoch. No test feature, label, ranking, or score enters this process.

Freeze the selected attention checkpoint, export Train/Validation [N,768]
embeddings and pass them to E2B-M4. Keep the representation frozen throughout
pair training. Do not refit on all development images.
Repeat with matching upstream seeds for the registered publication roster.

Final Test access requires the shared global lock after all E2A and E2B
branches and declared seeds are complete. M4 completion alone does not open
Final Test.

## 4. Attention diagnostics

Store validation attention weights and normalized entropy:

~~~text
normalized_entropy_i = -sum_j a_ij log(a_ij) / log(196)
~~~

Use a stable zero-log-zero convention; entropy is in [0,1].
Record effective support, entropy summaries and numerical validity.
Use deterministic example selection by sorted image ID within categories
declared before Final Test. Attention maps are descriptive aggregation
weights; they do not establish bird-part localization or causal importance.
No box/part annotation may influence training, checkpoint selection, or
ranking in this experiment.

If test attention plots are included, declare them in the global lock and
generate them as final diagnostics. They cannot trigger retraining.

## 5. Implementation and artifacts

Add configs/cub_e2a_m4.yaml and AttentionPatchPooling in
models/representations.py. Extend shared data/feature_cache.py for patch
shards and training/representation.py for the M4 module. Reuse the common
extract/train/evaluate/run scripts and evaluator.

Output root: outputs/e2a_attention_pool/<run_id>/seed_<seed>/.

Save config_resolved.yaml, environment.json, metadata.json, input patch/split
manifests, checkpoints/best.pt, training history/sampling/selection trace,
embeddings/{train,validation}.pt, embeddings/manifest.json,
rankings/validation_top100.pt, metrics/validation.json,
attention/{validation_weights,validation_entropy}.pt, and freeze.json.

Checkpoints contain attention/proxies, optimizer/scheduler, epoch/global step,
RNG/sampler states, validation hits, cache/split/config/code hashes and
parameter counts. Final Test artifacts are appended only by the locked final
campaign. Intermediate cache shards stay separate from portable run bundles
and are referenced with relative manifests and hashes.

## 6. Tests and acceptance

Test attention shape, positive weights summing to one, finite entropy bounds,
uniform scores reducing to M2 mean, and invariance to patch permutation.
Use CLS/register sentinel values to prove their exclusion. Reject empty or
wrong token dimensions, invalid raw embeddings and stale/mixed cache shards.

Verify hand-computed weighted pooling, initialization, parameter counts,
and attention/proxy gradients. The scalar output bias may have zero gradient;
that does not justify requiring nonzero gradients for every scalar parameter.

Compare single-device and two-rank global-batch loss and gradients for
attention weights and proxies. Test data split isolation, deterministic
resume, integer-hit epoch ties, checkpoint/cache mismatch, and final-lock
failure when any pair branch is missing.

Run source compilation and the complete pytest suite, including mocked CUB
and synthetic token integration tests. GPU parity tests must be run on the
target T4 runtime or explicitly recorded as pending.

Expected command after implementation:

~~~bash
torchrun --standalone --nproc_per_node=2 scripts/run_e2a.py --config configs/cub_e2a_m4.yaml --stage validation
~~~

Acceptance requires the selected checkpoint, development exports, diagnostic
artifacts and integrity tests. Every accepted M4 run continues into E2B-M4,
regardless of its relative validation rank.
