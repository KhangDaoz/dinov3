# E2A-M1: frozen CLS baseline plan

Protocol: e2ab-v2. Status: proposed implementation; no accepted metrics.

## 1. Method and scope

Apply [the shared protocol](e2a_plan.md) and
[the new specification](../experiments/e2a-e2b-new.md).
For image i, take the final normalized transformer CLS output c_i:

~~~text
raw_embedding_i = c_i                         # [768]
z_i = raw_embedding_i / ||raw_embedding_i||_2
cosine(i,j) = z_i dot z_j
~~~

M1 has zero trainable representation parameters. Use frozen DINOv3
ViT-B/16, 224×224 input, one CLS plus four register and 196 patch tokens.
CLS selection is independent of register/patch values. L2 normalization is
FP32; reject zero or nonfinite features.

M1 is the off-the-shelf baseline in all result tables and is one of four E2B
inputs. It is neither preselected as best nor required to improve another
method. Method/model provenance is recorded in [sources](sources.md).

## 2. Inputs and cache

Consume the common development token cache, indexed by image ID, with the
shared model/processor/precision/split hashes. Expected development CLS shape
is [5864,768]; Train and Validation views contain 4,687 and 1,177 rows.
A portable M1 manifest may reference the common CLS tensor without duplication.

Extract only development images before the joint E2A/E2B lock.
Do not accept an old metric, claimed baseline score, or E2A winner artifact
as an implementation oracle. Any reusable legacy feature data must pass the
shared cache checks and be isolated to development rows.

## 3. Implementation steps

1. Implement M1 fields in configs/cub_e2a_m1.yaml and config_e2a.py:
   method m1, source final_cls, dimension 768, L2 retrieval, shared model,
   processor, split, precision, and evaluator settings.
2. Reject projection/attention/training fields for this fixed representation.
3. Add CLSRepresentation in models/representations.py, accepting a typed
   token object and returning [B,768]. Keep raw and L2-normalized artifacts
   explicitly distinguished.
4. Implement the shared feature-cache adapter and provenance validation.
5. Run development validation with the common exact cosine evaluator.
   Verify serialized rankings reproduce integer Hits@1/2/4/8 exactly.
6. Export both Train and Validation embeddings for E2B-M1 and write
   freeze.json containing their hashes and the validation acceptance record.

Expected files, created only with working behavior:

~~~text
configs/cub_e2a_m1.yaml
src/uncertainty_retrieval/config_e2a.py
src/uncertainty_retrieval/data/{cub,feature_cache}.py
src/uncertainty_retrieval/models/{dinov3,representations}.py
src/uncertainty_retrieval/evaluation/representation.py
scripts/{extract_e2a_features,evaluate_e2a,run_e2a}.py
tests/unit/test_{config_e2a,representations,feature_cache,representation_evaluation}.py
tests/integration/test_e2a_m1_smoke.py
~~~

## 4. Artifacts and final evaluation

Output root: outputs/e2a_cls/<run_id>/seed_42/.

Save config_resolved.yaml, environment.json, metadata.json, input/split
manifests, embeddings/{train,validation}.pt or immutable references,
rankings/validation_top100.pt, metrics/validation.json, and freeze.json.
There is no learned M1 checkpoint or training history to invent.

The final runner verifies the global lock for all E2A/E2B branches and
planned seeds before extracting M1 test CLS or producing test metrics.
Final Test has 5,924 query/gallery images and must exclude each self-match.
No standalone M1 acceptance criterion depends on test performance.

M1 is deterministic given the cache and evaluator. Repeating the same feature
extraction or changing an unused training seed does not estimate training
variance. Its E2B pair heads do have training-seed variance.

## 5. Tests and acceptance

Test [B,768] selection, insensitivity to other tokens, frozen backbone and
zero optimized parameters; reject incompatible fields, checkpoint/processor
mismatches, duplicate/missing IDs, label drift, zero vectors and NaN/Inf.
Test split isolation, stable ties, self-match by ID under shuffled gallery,
correct denominators, serialization, and hand-computed Hits@K.
Verify CLS equals pooler_output and comes from post-final-LayerNorm
last_hidden_state; a mocked pre-norm hidden_states[-1] must not be accepted
as the source. Require eval mode as well as frozen parameters.

A mocked CUB-style sample must exercise RGB decoding and corrupt-image failure.
A synthetic token fixture must contain 1 CLS + 4 register + 196 patch tokens.
Distributed extraction must merge non-padding shards without missing IDs;
GPU agreement is checked on the target runtime using declared tolerances.

Run compilation and the complete pytest suite after implementation. Then:

~~~bash
torchrun --standalone --nproc_per_node=2 scripts/run_e2a.py --config configs/cub_e2a_m1.yaml --stage validation
~~~

This command is a proposed interface. Acceptance requires all development
integrity tests and complete artifacts, then authorizes handoff to E2B-M1.
It does not authorize early test evaluation.
