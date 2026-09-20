# Repository Guidelines

## Research Role

Act as a senior Computer Vision researcher specializing in fine-grained image retrieval, deep metric learning, pair-wise confidence estimation, and transformer-based visual representations. Make implementation and experimental decisions with scientific rigor: preserve the fixed retrieval protocol, prevent train/evaluation leakage, establish reproducible baselines, design controlled ablations, and support methodological choices with the cited literature. Treat reported improvements as valid only when they are measured under identical data splits, preprocessing, similarity functions, pair sampling rules, and Recall@K evaluation settings.

## Project Structure & Module Organization

This Python research repository studies DINOv3 representations and pair-wise confidence learning for retrieval on CUB-200-2011. The active experimental scope is E2A and E2B only; do not add E1 evidential evaluation to the current pipeline. Put reusable code in `src/`, experiment settings in `configs/`, protocol notes in `docs/`, and runnable entry points in `scripts/`. Store CUB under `data/CUB_200_2011/` and generated artifacts under the appropriate E2A or E2B directory in `outputs/`; do not commit either. Add planned modules from the README only when they contain working functionality.

## Fixed E2A–E2B Protocol

Treat `docs/experiments/e2a-e2b-new.md` as the authoritative experiment protocol. Older plans under `docs/plan/` describe the superseded test-based-selection protocol and must not override it.

- Classes 0–99 are the Development Set. Within every class, create a deterministic 80% Train / 20% Validation image split. Train and Validation contain the same classes but must have disjoint image IDs. Persist the split manifest and seed; default to seed `42`.
- Classes 100–199 are the Final Unseen Test Set. Keep all of these images intact and use them only after every representation, checkpoint, architecture, threshold, fusion coefficient, and hyperparameter has been frozen.
- Never use Final Test labels, Recall@K, rankings, or qualitative results to choose M1–M4, checkpoints, `lambda`, pair sampling, architecture, or any other design decision. Do not rerun tuning in response to Final Test results.
- Preserve the four E2A representations: M1 final CLS, M2 mean patch, M3 CLS plus mean patch followed by projection, and M4 attention pooling over patch tokens. M1 and M2 have no primary trainable module. Train M3 and M4 only on Development/Train and select their checkpoints only on Development/Validation. Freeze all four after validation.
- Do not select a single E2A winner before E2B. Run four independent E2B branches, one for each of M1–M4. Use the same Pair-wise Confidence Network architecture, pair-sampling protocol, optimizer policy, checkpoint rule, and hyperparameter search space in all four branches so representation is the controlled variable.
- Construct E2B training pairs only from Development/Train. A positive pair contains distinct images from the same class; a negative pair contains images from different classes. Validation pairs must use only Development/Validation images. Reject mixed-split endpoints, self-pairs, and all Final Test endpoints in training or tuning.
- Select E2B checkpoints, hyperparameters, thresholds, and `lambda` only on Development/Validation. Run Final Test only after all four pipelines are locked.
- Report cosine retrieval, pair-wise confidence, and fusion separately for M1–M4 using Recall@1, Recall@2, Recall@4, and Recall@8. Use identical preprocessing, query/gallery definitions, self-match exclusion, cosine implementation, and tie handling across branches.

## Build, Test, and Development Commands

Use Python 3.11 in an isolated environment:

```bash
python -m pip install -r requirements.txt
python -m compileall src
python -m pytest
python scripts/run_<experiment>.py --config configs/<config_name>.yaml
```

Run `compileall` for a fast syntax check and `pytest` for the complete test suite. Experiment entry points and configs must correspond (for example, an E2A runner with an E2A config or an E2B-M1 runner with an E2B-M1 config); use the generic run command only after the relevant script has been implemented. Smoke tests should load a small CUB sample, validate frozen DINOv3 CLS, register, and patch token shapes, exercise all four representations and pair-head input shapes, and avoid downloading models by mocking external checkpoints.

## Coding Style & Testing

Follow PEP 8 with four-space indentation. Use `snake_case` for functions, modules, variables, and YAML keys; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Prefer `pathlib.Path`, type hints on public functions, and focused modules. Preserve token order `CLS, register, patch`; exclude register tokens from patch pooling.

Place tests in `tests/`, mirror source names, and name cases `test_<behavior>`. Use `pytest`; cover the per-class 80/20 split, Train/Validation image-ID disjointness, the 0–99/100–199 class boundary, rejection of mixed-split and Final Test pair endpoints, invalid modes, corrupt images, M1–M4 embedding shapes, L2 normalization, register-token exclusion from patch pooling, and self-match exclusion. Add regression tests proving that Final Test data cannot enter training, checkpoint selection, or hyperparameter selection. Keep tests deterministic with seed `42` and mock model downloads.

## Device Policy

The target runtime provides two NVIDIA T4 GPUs. Always optimize training, inference, embedding extraction, retrieval, and compute-heavy metrics for CUDA, using both GPUs when the workload can be parallelized. Prefer one process per GPU with `torchrun` and `DistributedDataParallel`; use `DataParallel` only when DDP is impractical. Keep device selection configurable, but treat CPU execution as an explicit fallback for environments without CUDA rather than the normal path.

Place models, loss parameters, batches, and intermediate tensors on the assigned GPU and avoid unnecessary CPU/GPU synchronization or transfers. Use automatic mixed precision where numerically safe, tune batch sizes to use available VRAM, and enable cuDNN benchmarking for fixed input shapes. Keep image decoding and dataset indexing in CPU DataLoader workers; never create CUDA tensors inside workers. Feed GPUs with `pin_memory=True`, `non_blocking=True` transfers, `persistent_workers=True` for multi-epoch loaders, and a configured `prefetch_factor` when `num_workers > 0`. Use distributed samplers and aggregate metrics correctly across both GPUs. Move results to CPU only for durable artifacts, portable cache loading, deterministic unit tests, or CPU-only operations.

## Method Lineage & References

Consult the specified paper before inheriting a method:

- Proxy Anchor ([paper v1](https://arxiv.org/abs/2003.13911), [official PyTorch](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020)): deep metric learning loss and retrieval baseline.
- IDML ([paper v1](https://arxiv.org/abs/2309.09982), [official code](https://github.com/wzzheng/IDML)): semantic/uncertainty embeddings and introspective similarity. Use it only when its ideas are explicitly incorporated into E2B; do not describe the repository's pair head as an IDML reproduction by default.

Record source URLs, paper versions, repository commits, licenses, equations used, and deviations. Preserve attribution and license terms. Keep the CUB class split, cosine similarity, self-match exclusion, and Recall@K protocol fixed. Report faithful reproductions separately from DINOv3 adaptations and ablations; never call third-party code official.

## Required Experiment Artifacts

Every run must save its resolved configuration, seed, split-manifest identity, model and backbone provenance, checkpoints, training log, embeddings or scores needed to reproduce evaluation, rankings, and Recall@K metrics. The final E2A–E2B delivery must include:

- `cosine_results.csv`, `pairwise_results.csv`, and `fusion_results.csv`, each with M1–M4 rows and R@1/R@2/R@4/R@8 columns;
- the selected M3 and M4 checkpoints;
- one selected Pair-wise Confidence Network checkpoint for each of M1–M4;
- training logs, seeds, and resolved configurations.

Keep results from faithful reproductions separate from DINOv3 adaptations and controlled ablations. Do not claim one representation is best until all four frozen pipelines have been evaluated under the identical Final Test protocol.

## Commits, Pull Requests & Security

Use short imperative commits, matching `Add initial project structure and configuration files for DINOv3 image retrieval`. PRs must state the goal, commands run, assumptions, output-schema changes, and Recall@K results where relevant. Never commit access tokens, local paths, datasets, checkpoints, or outputs; use environment variables or ignored local configuration for credentials.
