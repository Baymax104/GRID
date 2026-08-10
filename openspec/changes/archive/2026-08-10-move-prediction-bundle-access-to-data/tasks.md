## 1. Data Model Relocation

- [x] 1.1 Delete unused `ItemTextBatch` from `src/data/components/data_models.py`.
- [x] 1.2 Move `ModelOutput` into `src/data/components/data_models.py`.
- [x] 1.3 Update all Python imports from `src.inference.model_output` to `src.data.components.data_models`.
- [x] 1.4 Delete `src/inference/model_output.py` after all references are removed.

## 2. Bundle Utility Relocation

- [x] 2.1 Move `load_model_output`, `load_semantic_id_tensor`, and `gather_predictions_by_keys` into `src/data/utils.py`.
- [x] 2.2 Update all Python imports from `src.inference.utils` to `src.data.utils`.
- [x] 2.3 Update all Hydra `_target_` strings from `src.inference.utils.*` to `src.data.utils.*`.
- [x] 2.4 Delete `src/inference/utils.py` after all references are removed.

## 3. Specs and Tests

- [x] 3.1 Update living specs for data model role separation, prediction output protocol, and keyed prediction bundle artifact.
- [x] 3.2 Update prediction writer tests to import `ModelOutput` from data models.
- [x] 3.3 Add or update focused tests for data utility loading, semantic ID extraction, and key lookup.
- [x] 3.4 Add or update residual path tests or scans for removed `src.inference.model_output`, `src.inference.utils`, and `ItemTextBatch`.

## 4. Validation

- [x] 4.1 Run focused pytest for inference writer and data utility coverage.
- [x] 4.2 Run scoped ruff checks for touched source and tests.
- [x] 4.3 Run residual scans for old imports and Hydra targets.
- [x] 4.4 Run `openspec validate move-prediction-bundle-access-to-data --strict`.
