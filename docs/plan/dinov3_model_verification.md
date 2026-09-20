# DINOv3 checkpoint and processor verification

Verified on 2026-09-20 for protocol e2ab-v2 using user-authorized access.
Scope: authenticated repository metadata/files, local processor behavior
on synthetic images, and model structure on the PyTorch meta device.
No CUB images, retrieval scores, pretrained weight loading, or GPU runs
were used in this audit. Credentials were not persisted in project files.

## 1. Repository identity and file integrity

Model: facebook/dinov3-vitb16-pretrain-lvd1689m.
The authenticated API resolved the requested revision exactly to
5931719e67bbdb9737e363e781fb0c67687896bc.
The repository is public with manual gated access, uses Transformers, and
declares the image-feature-extraction task.

Primary metadata sources at that immutable revision:

- [Model API](https://huggingface.co/api/models/facebook/dinov3-vitb16-pretrain-lvd1689m/revision/5931719e67bbdb9737e363e781fb0c67687896bc?blobs=true)
- [Model configuration](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/blob/5931719e67bbdb9737e363e781fb0c67687896bc/config.json)
- [Processor configuration](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/blob/5931719e67bbdb9737e363e781fb0c67687896bc/preprocessor_config.json)
- [Model card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/blob/5931719e67bbdb9737e363e781fb0c67687896bc/README.md)
- [License](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/blob/5931719e67bbdb9737e363e781fb0c67687896bc/LICENSE.md)

| File | Bytes | SHA-256 | Verification type |
|---|---:|---|---|
| config.json | 744 | 3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec | Computed from downloaded bytes |
| preprocessor_config.json | 585 | 960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7 | Computed from downloaded bytes |
| README.md | 14,520 | 42990cce5247479866ce2e6aa375eb3c3d1717966c41b74890d377a31ad0b0fe | Computed from downloaded bytes |
| LICENSE.md | 7,503 | 25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e | Computed from downloaded bytes |
| model.safetensors | 342,662,192 | 9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b | Hub-reported LFS digest; full weight bytes not downloaded |

The weights have a verified remote identity, not a locally recomputed
content digest. Compare the downloaded weight file against the expected
SHA-256 before actual feature extraction. Do not confuse an LFS pointer
blob ID with the SHA-256 of model.safetensors.

## 2. Architecture contract

These values come from the pinned config and the Hub tensor inventory.

| Property | Value |
|---|---|
| Architecture / model type | DINOv3ViTModel / dinov3_vit |
| Input / patch size | 3 channels, nominal 224×224 / 16×16 |
| Hidden / intermediate dimension | 768 / 3072 |
| Transformer layers / attention heads | 12 / 12 (64 dimensions per head) |
| Register tokens | 4 |
| MLP | GELU, use_gated_mlp=false, mlp_bias=true |
| Attention biases | query=true, key=false, value=true, output projection=true |
| LayerNorm epsilon / LayerScale initial value | 1e-5 / 1.0 |
| Attention dropout / drop path | 0.0 / 0.0 |
| RoPE theta | 100.0 |
| Position augmentation config | shift=null, jitter=null, rescale=2.0 |
| Serialized weight dtype | float32 |
| Stored parameter count | 85,660,416, including special-token parameters |
| Serializer's Transformers version | 4.56.0.dev0 |

The serializer version identifies file provenance; it is not a tested
runtime lock or a requirement to install that development build.

Set both model.eval() and requires_grad_(False). Disabling gradients alone
does not disable training-mode position augmentation. The inspected model
applies position-coordinate rescaling only while training.

## 3. Exact preprocessing contract

The pinned processor declares DINOv3ViTImageProcessorFast. In the audited
Transformers 5.16.1 installation it resolves to DINOv3ViTImageProcessor with
the torchvision backend.

Use these operations in order:

1. Decode using the locked Pillow implementation and explicitly convert
   the image to RGB in the dataset loader. do_convert_rgb is null in the
   checkpoint and does not guarantee automatic conversion.
2. Convert uint8 pixels to a channels-first tensor.
3. Rescale to FP32 using 1/255 (0.00392156862745098).
4. Resize the whole tensor to height=224, width=224 using bilinear
   interpolation (resample=2) and antialias=True.
5. Normalize channels with mean [0.485,0.456,0.406] and
   std [0.229,0.224,0.225].
6. Return pixel_values in [B,3,224,224], FP32 before backbone AMP.

This is a direct square resize, including aspect-ratio deformation for
rectangular inputs. No shortest-edge resize or center crop is enabled:
do_center_crop=null and crop_size=null are inactive in the inspected
implementation. No project resolution override is needed at this revision.

The rescale-before-resize order and antialias setting were confirmed in
the model-specific implementation and by a synthetic numerical comparison;
they cannot be inferred from the JSON flags alone.
Do not substitute a generic resize-on-uint8/PIL, then rescale pipeline, or
another processor backend without checking numerical equivalence.

All M1–M4 branches use this same preprocessing. No random image augmentation,
bounding-box/part crop, automatic EXIF transpose or alternate normalization
may be introduced without an explicitly documented common protocol change.
Decode/RGB/indexing remain CPU-worker operations. The locked processor may
run on CPU or the assigned GPU; workers must not create CUDA tensors, and
CPU/GPU processor parity must be checked on the target runtime.

## 4. Token output and normalization contract

For 224×224 input, obtain output.last_hidden_state and slice:

~~~text
sequence = outputs.last_hidden_state     # [B,201,768], after final LayerNorm
cls = sequence[:, 0, :]                  # [B,768]
register = sequence[:, 1:5, :]           # [B,4,768]
patch = sequence[:, 5:, :]               # [B,196,768]
~~~

The embedding module concatenates CLS, registers, then flattened patches.
Patch order follows the 14×14 convolution grid with width varying fastest.
The inspected forward applies model.norm to the final encoder sequence,
then returns that sequence as last_hidden_state. pooler_output equals its
CLS slice; it is not an average over tokens.

With output_hidden_states enabled, hidden_states[-1] comes from the encoder
before the final model.norm in this backend. Use last_hidden_state directly;
do not pool pre-norm tokens or apply the final LayerNorm twice.

This LayerNorm is distinct from retrieval L2 normalization. M1–M4 apply
their specified representation construction before the common FP32 L2 step.
No register token is part of the M2 mean, M3 patch summary or M4 attention.

## 5. Offline checks actually run

Audit environment, not the target training environment:

| Component | Observed version |
|---|---|
| Python | 3.14.7 |
| PyTorch / torchvision | 2.14.0+cu130 / 0.29.0+cu130 |
| Transformers / huggingface-hub | 5.16.1 / 1.30.0 |
| Pillow / NumPy | 12.3.0 / 2.4.6 |
| CUDA availability | false |

The project runtime remains Python 3.11 and two NVIDIA T4 GPUs.
Do not copy this local CUDA/Python environment into the target config
without checking compatibility and numerical behavior.

A synthetic uint8 RGB image of shape [137,301,3], generated by
NumPy default_rng(42).integers(0,256,...,dtype=uint8), produced
[1,3,224,224] FP32 output. The result matched a separately assembled
torchvision reference (rescale -> bilinear antialiased resize -> normalize)
with maximum absolute error 0.0; repeated processing was bitwise equal.
Explicit RGB conversion also passed for synthetic L and RGBA inputs.

Synthetic output bytes SHA-256 on this environment:
517b3a9c1dd0ea311bf10e8b3db7c29dfb0a9ff4cb4777cd16f97ebf29f0dd9a.
This is a backend-specific regression record, not a cross-platform
bitwise-equality requirement or a CUB embedding hash.

A model instantiated from the config on the meta device had 85,660,416
parameters and zero trainable parameters after freezing.
Shape-only forward returned [1,201,768], pooler [1,768], registers [1,4,768]
and patches [1,196,768]. No pretrained tensors were loaded, and a meta
forward cannot validate their numerical output.

Inspected installed source SHA-256:

- models/dinov3_vit/image_processing_dinov3_vit.py:
  04efcf5edbbd64dd0acad29201af66c3680770b6c10bc5faedd0be8ca3b11720
- models/dinov3_vit/modeling_dinov3_vit.py:
  e01b5da28fe5f167e3d57a05121007460ed3fc5d5d91a19dc51c7d6772dcb916

## 6. License and remaining runtime gates

The pinned LICENSE.md is the DINOv3 License, last updated August 19, 2025.
Record its hash and the model attribution in experiment provenance.
Sections 1.b.i–ii cover providing the agreement with redistributed DINO
materials/derivatives and acknowledging DINO use in publications.
Use the full license as the source of terms; it is separate from the
Apache-2.0 headers on the inspected Transformers implementation.

The earlier unauthenticated HTTP 401 is resolved for this authorized audit.
Metadata, processor settings, license bytes and structural shape checks pass.
The following remain execution gates in [the shared plan](e2a_plan.md):

- Download and hash-check the actual pretrained weights on the runtime.
- Lock a compatible Python 3.11/T4 software stack and rerun processor checks.
- Verify real-weight CLS/register/patch outputs, final-LayerNorm semantics
  and CPU/GPU processor parity on development-only inputs.
- Run the common FP32-versus-AMP fidelity comparison and two-GPU shard checks.
- Complete leakage, training, metric and portable-export acceptance tests.

No Final Test access or scientific performance claim is justified by this
metadata audit alone.
