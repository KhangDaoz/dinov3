# Sources, versions, and method lineage

Checked on 2026-09-20 for protocol e2ab-v2.
Links below identify primary sources. Reading a source does not establish
that its method has been reproduced or that this project has measured gains.

## 1. Sources used by the active experiment

| Source | Verified evidence | Use in this project |
|---|---|---|
| [Proxy Anchor paper v1](https://arxiv.org/pdf/2003.13911v1) | Eq. (4), positive/all-proxy denominators; alpha=32, delta=0.1 in Section 4.2 | Shared M3/M4 metric objective |
| [Author Proxy Anchor code](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020/tree/51db57031e38f75c03f69bbdfad1a3233afd9787) | Commit 51db57031e38f75c03f69bbdfad1a3233afd9787; author repository | Numerical reference for the loss |
| [Loss implementation](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020/blob/51db57031e38f75c03f69bbdfad1a3233afd9787/code/losses.py) | Normalized cosine, alpha/margin signs, Kaiming-normal proxies | Cross-check equations and initialization |
| [Proxy Anchor license](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020/blob/51db57031e38f75c03f69bbdfad1a3233afd9787/LICENSE) | MIT, copyright 2020 Sungyeon Kim | Retain notices for any copied/adapted substantial code |
| [DINOv3 paper v1](https://arxiv.org/abs/2508.10104v1) | Model family source | Frozen visual backbone lineage |
| [Meta model card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | ViT-B/16: dimension 768, four registers; DINOv3 License | Model architecture and access/provenance |
| [CUB project page](https://www.vision.caltech.edu/datasets/cub_200_2011/) | 200 classes, 11,788 images, dataset citation and usage notice | Dataset attribution and scope |

Proxy Anchor reproduction boundaries: preserve the loss equation but change
the backbone to frozen DINOv3, inputs to cached deterministic features,
output dimension to 768, representation heads, training sampler/budget and
optimizer policy. These are project adaptations. Log-sum-exp implements the
same loss stably. The author code's hardcoded CUDA allocation and CPU label
conversion are not requirements for the new implementation.

The inspected loss body and its defaults were checked against the paper;
no third-party source was copied into this repository during this plan edit.

## 2. User-provided related work, outside active method implementation

- [IDML v1](https://arxiv.org/html/2309.09982v1) learns semantic and uncertainty
  embeddings with introspective similarity. It motivates careful treatment
  of ambiguous pairs, but the present BCE MLP and raw-score fusion do not
  implement that metric. The [author repository at inspected commit](https://github.com/wzzheng/IDML/tree/c49aaa304b7646d1e086107d216ccda34c3b16f2)
  is c49aaa304b7646d1e086107d216ccda34c3b16f2. Its inspected root tree has no
  top-level license file; reuse permission is unverified. No IDML code is
  imported by this plan.
- [EDL v3](https://arxiv.org/abs/1806.01768v3) and the
  [author notebook](https://muratsensoy.github.io/uncertainty.html) concern
  Dirichlet classification evidence. They are historical context; no EDL
  head or evaluation is required by the current E2A/E2B run.
- [Evidential Transformers v2](https://arxiv.org/html/2409.01082v2), Sections
  2.2–2.4, describes alpha embeddings with L2 distance, uncertainty
  reranking and distribution distances. These are separate methods from
  the project pair head. They are not included as active comparison rows.
  No author code is reused; code/license provenance is not asserted.

The M3 affine fusion, M4 scorer, E2B MLP architecture, 16/8/8 sampling,
lambda grid and selection rules are project specifications inherited or
resolved from the local plans. They must not be attributed to IDML, EDL,
or Evidential Transformers as published formulas.

## 3. Authenticated backbone verification and runtime boundary

Authorized access resolved the revision exactly to
5931719e67bbdb9737e363e781fb0c67687896bc, superseding the earlier HTTP 401.
Downloaded and hashed config.json, preprocessor_config.json, README.md and
LICENSE.md. The Hub also reports the weight-file size and LFS SHA-256;
the full weight file has not been downloaded or locally hash-verified.
See [the model verification record](dinov3_model_verification.md) for exact
values and evidence.

The processor specifies direct 224×224 bilinear resize. Source inspection
and synthetic checks establish rescale-before-resize, antialias=True,
ImageNet mean/std and inactive center crop. RGB conversion is explicit in
the loader. No resolution override is needed. Meta-device forward confirms
201-token shapes; actual pretrained numerical outputs and T4/AMP parity
remain runtime acceptance gates.

The pinned license file is the DINOv3 License dated August 19, 2025;
preserve its terms and attribution in release provenance. The audited
local library versions differ from the required Python 3.11/T4 runtime and
are recorded as verification context, not a target environment lock.
The CUB page limits image use to noncommercial research/education and warns
of overlap with ImageNet/Flickr pretraining sources. This does not establish
whether DINOv3 saw these particular images. Record overlap as unknown unless
verified; publish dataset access instructions rather than bundling images.

## 4. Research reporting references

[NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist)
covers claim scope, limitations, reproducibility, experiment settings,
uncertainty reporting, compute and asset attribution. It asks authors to
explain what their error bars measure. It does not prescribe this project's
seed count, candidate budget or bootstrap procedure.

[CVPR 2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)
provide venue-specific submission, supplementary-material and research
conduct requirements. Recheck the actual target venue/year before submission.

These are reference practices, not a universal certification or a guarantee
of acceptance. [The verification plan](research_verification.md) translates
them into project-specific evidence gates.

## 5. Provenance record required at execution

For each reused asset record author/title, exact paper version, URL,
repository/model commit if applicable, license/terms, file hashes, equation
or component used, implementation changes, access date, and backend/package
versions. Distinguish verified entries from unavailable ones. Do not copy
unlicensed source, publish credentials, or label an unaffiliated fork official.
