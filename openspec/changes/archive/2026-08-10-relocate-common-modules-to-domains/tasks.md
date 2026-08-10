## 1. Embedding Module Relocation

- [x] 1.1 Move `EmbeddingAggregator` to `src/embedding/embedding_aggregator.py`.
- [x] 1.2 Update `src/embedding/hf_language_model.py` to import the embedding-local module.
- [x] 1.3 Update `configs/model/sem_embeds_inference.yaml` to use the new Hydra `_target_`.

## 2. RQVAE Module Relocation

- [x] 2.1 Move `MLP` to `src/quantization/rqvae/mlp.py`.
- [x] 2.2 Move `NormalizeLayer` to `src/quantization/rqvae/normalize_layer.py`.
- [x] 2.3 Update `configs/model/rqvae_train.yaml` to use the new Hydra `_target_` paths.

## 3. Cleanup and Specs

- [x] 3.1 Delete `src/common/modules` after all official references are removed.
- [x] 3.2 Update living specs for module path alignment and flattened common modules.
- [x] 3.3 Add or update focused tests for config path declarations and moved module imports.
- [x] 3.4 Run residual scans for `src.common.modules` and `common/modules`.

## 4. Validation

- [x] 4.1 Run focused tests for embedding, RQVAE config, and affected module path assertions.
- [x] 4.2 Run scoped ruff checks for moved modules and touched tests.
- [x] 4.3 Run `openspec validate relocate-common-modules-to-domains --strict`.
