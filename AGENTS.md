# Repository Guidelines

## Project Structure & Module Organization

This Python research repository studies uncertainty-aware DINOv3 retrieval on CUB-200-2011. Put reusable code in `src/`, experiment settings in `configs/`, protocol notes in `docs/`, and runnable entry points in `scripts/`. Store CUB under `data/CUB_200_2011/` and generated artifacts under `outputs/<representation>/`; do not commit either. Add planned modules from the README only when they contain working functionality.

## Build, Test, and Development Commands

Use Python 3.11 in an isolated environment:

```bash
python -m pip install -r requirements.txt
python src/dinov3_backbone.py --config configs/cub_e2a.yaml --root data
python -m compileall src
```

The smoke check loads one CUB image and validates frozen DINOv3 CLS, register, and patch tokens.

## Coding Style & Testing

Follow PEP 8 with four-space indentation. Use `snake_case` for functions, modules, variables, and YAML keys; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Prefer `pathlib.Path`, type hints on public functions, and focused modules. Preserve token order `CLS, register, patch`; exclude register tokens from patch pooling.

Place tests in `tests/`, mirror source names, and name cases `test_<behavior>`. Use `pytest`; cover split boundaries, invalid modes, corrupt images, embedding shapes, L2 normalization, and self-match exclusion. Keep tests deterministic with seed `42` and mock model downloads.

## Device Policy

Use the resolved `device` throughout model inference, tensor operations, and retrieval. With `device: auto`, prefer CUDA whenever available and fall back to CPU only when no GPU exists. Do not hard-code CPU for compute paths; CPU is appropriate for persisted embeddings, portable cache loading, and deterministic unit tests.

## Method Lineage & References

Consult the specified paper before inheriting a method:

- EDL ([paper v3](https://arxiv.org/abs/1806.01768), [author notebook](https://muratsensoy.github.io/uncertainty.html)): Dirichlet evidence and uncertainty.
- Proxy Anchor ([paper v1](https://arxiv.org/abs/2003.13911), [official PyTorch](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020)): loss and retrieval baseline.
- IDML ([paper v1](https://arxiv.org/abs/2309.09982), [official code](https://github.com/wzzheng/IDML)): semantic/uncertainty embeddings and introspective similarity.
- Evidential Transformers ([paper v2](https://arxiv.org/abs/2409.01082)): transformer retrieval; no author-linked public code was found as of September 2026.

Record source URLs, paper versions, repository commits, licenses, equations used, and deviations. Preserve attribution and license terms. Keep the CUB class split, cosine similarity, self-match exclusion, and Recall@K protocol fixed. Report faithful reproductions separately from DINOv3 adaptations and ablations; never call third-party code official.

## Commits, Pull Requests & Security

Use short imperative commits, matching `Add initial project structure and configuration files for DINOv3 image retrieval`. PRs must state the goal, commands run, assumptions, output-schema changes, and Recall@K results where relevant. Never commit access tokens, local paths, datasets, checkpoints, or outputs; use environment variables or ignored local configuration for credentials.
