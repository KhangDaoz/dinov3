# E1 Implementation and Experiment Plan

## 1. Research question and scope

E1 tests whether image-level evidential uncertainty can improve frozen DINOv3 retrieval on CUB-200-2011. The primary comparison is:

- **B0 — Retrieval baseline:** frozen DINOv3 ViT-B/16 final CLS token, L2 normalization, cosine similarity.
- **E1 — Proposed:** the identical B0 candidate list, followed by uncertainty-aware reranking from a separately trained Evidential Head.

The backbone remains frozen. E1 is a **DINOv3 adaptation**, not a faithful reproduction of the GC ViT/DeiT experiments in Evidential Transformers. Do not claim that image uncertainty models pairwise correctness; that limitation is the motivation for E2B.

**Evidential embedding**—using the Dirichlet parameter vector \(\boldsymbol{\alpha}\) as a retrieval embedding—is explicitly outside E1 and belongs to E2. E1 retrieves only with the normalized DINOv3 CLS feature and uses evidential outputs solely for image-level uncertainty analysis and reranking. Nevertheless, E1 must persist the complete evidential outputs so E2 can reuse them without recomputation.

## 2. Method lineage

Use the following sources and record their versions in every report:

- Sensoy, Kaplan, Kandemir, [Evidential Deep Learning to Quantify Classification Uncertainty, arXiv:1806.01768v3](https://arxiv.org/abs/1806.01768v3), NeurIPS 2018. Use non-negative evidence, Dirichlet parameters, subjective uncertainty, expected squared-error Bayes risk, and annealed KL regularization.
- Đorđević and Kumar, [Evidential Transformers for Improved Image Retrieval, arXiv:2409.01082v2](https://arxiv.org/abs/2409.01082v2), CC BY 4.0. Adapt its top-N uncertainty reranking while replacing its trained GC ViT/DeiT backbone with frozen DINOv3.

No author-linked implementation is assumed. Record the exact equations implemented, source URLs, licenses, deviations, dependency versions, DINOv3 checkpoint identifier/revision, and repository commit.

## 3. Fixed protocol and leakage prevention

- Dataset root: `data/CUB_200_2011/`.
- Development classes: original labels 0–99 (5,864 images currently expected).
- Final test classes: original labels 100–199 (5,924 images currently expected).
- Ignore `train_test_split.txt` for this class-disjoint retrieval protocol.
- Use the same RGB processor/checkpoint for every method.
- Query and gallery are the same evaluation set; mask each query's own image before ranking.
- Report Recall@1, @2, @4, and @8.
- Seed Python, NumPy, PyTorch, samplers, and DataLoader generators with `42`; enable deterministic evaluation.

Create one deterministic, stratified image-level development split within classes 0–99 (recommended 80% fit / 20% validation per class). Use it only to select architecture, epoch, top-N, and reranking strength. After selection, retrain the Evidential Head on all development images once and evaluate the untouched classes 100–199 once. Never use final-test labels for checkpoint selection, normalization, or tuning.

## 4. Model and mathematics

For a frozen CLS feature \(x \in \mathbb{R}^{768}\), the Evidential Head outputs logits for the 100 development classes. Convert logits to finite non-negative evidence using `softplus`:

\[
e_k=\operatorname{softplus}(f_k(x)),\quad
\alpha_k=e_k+1,\quad
S=\sum_{k=1}^{K}\alpha_k.
\]

Predicted class probability, belief, and image-level uncertainty are:

\[
\hat p_k=\alpha_k/S,\quad b_k=e_k/S,\quad u=K/S.
\]

Implement the EDL expected squared-error Bayes risk:

\[
\mathcal L_{data} =
\sum_k (y_k-\alpha_k/S)^2 +
\frac{\alpha_k(S-\alpha_k)}{S^2(S+1)}.
\]

Add the EDL KL divergence from the label-adjusted Dirichlet
\(\tilde\alpha=y+(1-y)\odot\alpha\) to \(Dir(\mathbf 1)\), with
\(\lambda_t=\min(1,t/T_{anneal})\). Compute `lgamma`, `digamma`, Dirichlet strength, KL loss, and uncertainty in FP32 even when the remaining forward pass uses AMP. Assert finite values and \(\alpha_k\ge1\), \(0<u\le1\).

Start with a minimal head `LayerNorm(768) -> Linear(768, 100)`. Treat a one-hidden-layer MLP as a prespecified ablation, not an automatic expansion. The retrieval embedding for the primary E1 comparison remains the unchanged normalized DINOv3 CLS feature so reranking is the only treatment.

## 5. Reranking variants and controls

Generate the full cosine ranking once, then alter only its first \(N\) non-self candidates.

1. **R0:** cosine only; no reranking.
2. **R1, paper-faithful adaptation:** reorder top-N candidates by ascending candidate uncertainty; break equal-uncertainty ties by original cosine rank.
3. **R2, score fusion ablation:** within each query's top-N, min-max normalize cosine and the image-level certainty score \(c_{cert}=1-u\) using fit/validation-defined handling for zero ranges, then use
   \[
   s'=(1-\beta)\tilde s_{cos}+\beta\tilde c_{cert}.
   \]
4. **R3, negative control:** randomly permute top-N with a fixed seed.

Search a small prespecified grid on validation only: \(N\in\{10,20,50,100\}\) and \(\beta\in\{0.1,0.25,0.5,0.75\}\). Query uncertainty is constant across candidates and therefore cannot change their order; do not include it in the primary score. Preserve the original order outside top-N.

## 6. Evaluation and analysis

Primary endpoint: change in Recall@1 of R1 versus R0 on the untouched final test set. Secondary endpoints are Recall@2/4/8 and R2.

Also report:

- Mean and standard deviation over seeds 42, 43, and 44 for learned-head results; cache B0 once.
- Per-query paired bootstrap 95% confidence intervals for Recall@K deltas.
- Top-1 correctness detection using uncertainty: AUROC, AUPRC, and risk-coverage/AURC.
- Uncertainty distributions for correct versus incorrect top-1 retrievals.
- Recall@K stratified by query-uncertainty quartile.
- Qualitative success/failure cases containing query, original top results, reranked results, labels, cosine scores, and uncertainty.

For correctness detection, define the query-level score before running analysis (for example, uncertainty of the original top-1 candidate). Do not change this definition after seeing final-test results. EDL class probabilities refer to development classes, so do not report final-test classification accuracy as a meaningful endpoint.

## 7. Implementation sequence

### Phase A — Foundation

- Add packaging and pinned runtime dependencies in `pyproject.toml` and `requirements.txt`.
- Implement configuration loading, resolved-config persistence, logging, seeding, device/DDP setup, and artifact metadata.
- Implement CUB manifest parsing, RGB validation, class remapping, deterministic development split, and corrupt-image errors.

### Phase B — Frozen DINOv3 baseline

- Implement DINOv3 loading with an explicit checkpoint/revision and frozen parameters.
- Extract tokens in order CLS, register, patch; cache embeddings on CPU with image IDs, original labels, split, checkpoint, preprocessing, dtype, and schema version.
- Implement chunked GPU cosine retrieval, diagonal masking, deterministic tie-breaking, and Recall@K.
- Run B0 and freeze its result before adding reranking.

### Phase C — Evidential learning

- Implement the Evidential Head, EDL loss, KL annealing, and uncertainty.
- Train only the head using cached CLS features first. Use one DDP process per T4, `DistributedSampler`, AMP for safe operations, pinned memory, non-blocking transfers, and rank-zero-only artifact writes.
- Select the checkpoint using a prespecified validation objective: lowest validation EDL loss, with Recall@1 used only as a reported secondary measure.
- Export the full per-image evidence vector \(\mathbf e\), Dirichlet parameter vector \(\boldsymbol\alpha\), and scalar uncertainty \(u\), together with class probabilities, prediction, image ID, original label, split, class ordering, dtype, and schema version. Do not retain only summaries such as maximum evidence. Store these outputs losslessly enough for E2 to reuse \(\mathbf e\), \(\boldsymbol\alpha\), and \(u\) without another model forward pass.

### Phase D — Reranking and reporting

- Implement R0–R3 from a shared immutable baseline ranking.
- Select N and beta on validation, lock them in the resolved config, retrain on all development data, and run the final test.
- Produce metrics JSON/CSV, paired confidence intervals, plots, and failure-case manifests.
- Write `reports/e1.tex` with separate tables for adaptation results and ablations.

## 8. Planned files

```text
configs/cub_e1.yaml
src/uncertainty_retrieval/config.py
src/uncertainty_retrieval/utils.py
src/uncertainty_retrieval/data/cub.py
src/uncertainty_retrieval/models/dinov3.py
src/uncertainty_retrieval/models/evidential.py
src/uncertainty_retrieval/training/evidential.py
src/uncertainty_retrieval/evaluation/retrieval.py
src/uncertainty_retrieval/evaluation/evidential.py
scripts/extract_features.py
scripts/train_evidential.py
scripts/evaluate.py
scripts/run_e1.py
tests/unit/test_cub.py
tests/unit/test_retrieval.py
tests/unit/test_evidential.py
tests/integration/test_feature_extraction.py
tests/integration/test_experiment_smoke.py
```

Add each file only when its corresponding phase has working behavior.

## 9. Configuration contract

`configs/cub_e1.yaml` should expose, at minimum:

- dataset root, split policy, validation fraction, and seed;
- DINOv3 model ID/revision, processor settings, token type, and cache path;
- head architecture, class count, evidence activation, loss, KL coefficient, and annealing epochs;
- optimizer, learning rate, weight decay, epochs, batch size, workers, prefetching, AMP, and DDP;
- retrieval metric, K values, self-match exclusion, similarity chunk size, top-N grid, beta grid, and tie policy;
- output directory, schema version, checkpoint rule, and artifact toggles.

Reject unknown keys, invalid modes, non-positive batch sizes, class-count mismatches, and configurations that include register tokens in patch pooling.

## 10. Test and acceptance gates

Unit tests must cover class boundaries/counts, stable development splits, corrupt images, evidence/alpha/uncertainty equations, KL against a trusted small example, loss gradients, finite AMP behavior, embedding shapes, L2 normalization, self-match exclusion, ties, top-N boundary preservation, and Recall@K on hand-computed data.

The integration smoke test must use a tiny fixture and mocked checkpoint download. It must verify frozen backbone parameters, token shapes/order, one optimizer step on head parameters only, cache round-trip, deterministic reranking, and artifact schemas.

Do not start the full E1 run until:

- `python -m compileall src` passes;
- `python -m pytest` passes;
- B0 Recall@K is reproducible across repeated evaluation;
- DDP and single-GPU evaluation agree within numerical tolerance;
- a small overfit test reduces EDL loss without NaN/Inf;
- test labels are inaccessible to tuning code.

## 11. Artifact layout

```text
outputs/e1_evidential/<run_id>/
├── config_resolved.yaml
├── environment.json
├── metadata.json
├── checkpoints/best.pt
├── embeddings/{development,validation,test}.pt
├── uncertainty/{development,validation,test}.parquet  # full e, alpha, and u
├── rankings/{baseline,rerank_uncertainty,rerank_fusion}.parquet
├── metrics/{recall,uncertainty,bootstrap}.json
├── figures/
└── failure_cases/
```

Metadata must include the Git commit, dirty-tree flag, command, seeds, checkpoint/revision, data-manifest hash, split IDs/hash, package versions, CUDA/cuDNN versions, GPU names, wall time, and deviations from the cited methods. Never commit this directory.

## 12. Decision rule

The primary E1 decision compares only the locked paper-faithful R1 configuration against R0. E1 supports the primary hypothesis only if R1 improves final-test Recall@1 over R0 and the paired 95% confidence interval for that delta excludes zero, without a material regression at Recall@2/4/8. Otherwise report a null or negative primary result. Treat R2 strictly as a secondary fusion ablation; it cannot make the primary E1 hypothesis successful. Independently report whether uncertainty predicts top-1 correctness; improved ranking and useful uncertainty are distinct claims.
