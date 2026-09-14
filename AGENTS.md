# Repository Guidelines

## Research Role

Act as a senior Computer Vision researcher specializing in fine-grained image retrieval, deep metric learning, uncertainty estimation, and transformer-based visual representations. Make implementation and experimental decisions with scientific rigor: preserve the fixed retrieval protocol, prevent train/evaluation leakage, establish reproducible baselines, design controlled ablations, and support methodological choices with the cited literature. Treat reported improvements as valid only when they are measured under identical data splits, preprocessing, similarity functions, and Recall@K evaluation settings.

## Project Structure & Module Organization

This Python research repository studies uncertainty-aware DINOv3 retrieval on CUB-200-2011. Put reusable code in `src/`, experiment settings in `configs/`, protocol notes in `docs/`, and runnable entry points in `scripts/`. Store CUB under `data/CUB_200_2011/` and generated artifacts under `outputs/<representation>/`; do not commit either. Add planned modules from the README only when they contain working functionality.

## Build, Test, and Development Commands

Use Python 3.11 in an isolated environment:

```bash
python -m pip install -r requirements.txt
python -m compileall src
python -m pytest
python scripts/run_<experiment>.py --config configs/<config_name>.yaml
```

Run `compileall` for a fast syntax check and `pytest` for the complete test suite. Experiment entry points and configs must correspond (for example, an E2A runner with an E2A config); use the generic run command only after the relevant script has been implemented. Smoke tests should load a small CUB sample, validate frozen DINOv3 CLS, register, and patch token shapes, and avoid downloading models by mocking external checkpoints.

## Coding Style & Testing

Follow PEP 8 with four-space indentation. Use `snake_case` for functions, modules, variables, and YAML keys; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Prefer `pathlib.Path`, type hints on public functions, and focused modules. Preserve token order `CLS, register, patch`; exclude register tokens from patch pooling.

Place tests in `tests/`, mirror source names, and name cases `test_<behavior>`. Use `pytest`; cover split boundaries, invalid modes, corrupt images, embedding shapes, L2 normalization, and self-match exclusion. Keep tests deterministic with seed `42` and mock model downloads.

## Device Policy

The target runtime provides two NVIDIA T4 GPUs. Always optimize training, inference, embedding extraction, retrieval, and compute-heavy metrics for CUDA, using both GPUs when the workload can be parallelized. Prefer one process per GPU with `torchrun` and `DistributedDataParallel`; use `DataParallel` only when DDP is impractical. Keep device selection configurable, but treat CPU execution as an explicit fallback for environments without CUDA rather than the normal path.

Place models, loss parameters, batches, and intermediate tensors on the assigned GPU and avoid unnecessary CPU/GPU synchronization or transfers. Use automatic mixed precision where numerically safe, tune batch sizes to use available VRAM, and enable cuDNN benchmarking for fixed input shapes. Keep image decoding and dataset indexing in CPU DataLoader workers; never create CUDA tensors inside workers. Feed GPUs with `pin_memory=True`, `non_blocking=True` transfers, `persistent_workers=True` for multi-epoch loaders, and a configured `prefetch_factor` when `num_workers > 0`. Use distributed samplers and aggregate metrics correctly across both GPUs. Move results to CPU only for durable artifacts, portable cache loading, deterministic unit tests, or CPU-only operations.

## Method Lineage & References

Consult the specified paper before inheriting a method:

- EDL ([paper v3](https://arxiv.org/abs/1806.01768), [author notebook](https://muratsensoy.github.io/uncertainty.html)): Dirichlet evidence and uncertainty.
- Proxy Anchor ([paper v1](https://arxiv.org/abs/2003.13911), [official PyTorch](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020)): loss and retrieval baseline.
- IDML ([paper v1](https://arxiv.org/abs/2309.09982), [official code](https://github.com/wzzheng/IDML)): semantic/uncertainty embeddings and introspective similarity.
- Evidential Transformers ([paper v2](https://arxiv.org/abs/2409.01082)): transformer retrieval; no author-linked public code was found as of September 2026.

Record source URLs, paper versions, repository commits, licenses, equations used, and deviations. Preserve attribution and license terms. Keep the CUB class split, cosine similarity, self-match exclusion, and Recall@K protocol fixed. Report faithful reproductions separately from DINOv3 adaptations and ablations; never call third-party code official.

## Commits, Pull Requests & Security

Use short imperative commits, matching `Add initial project structure and configuration files for DINOv3 image retrieval`. PRs must state the goal, commands run, assumptions, output-schema changes, and Recall@K results where relevant. Never commit access tokens, local paths, datasets, checkpoints, or outputs; use environment variables or ignored local configuration for credentials.
