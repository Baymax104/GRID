## 1. Common Config Package

- [x] 1.1 Add `src/common/configs/__init__.py`.
- [x] 1.2 Add `src/common/configs/data.py` with `DatasetConfig`, `SequenceDataloaderConfig`, and `ItemDataloaderConfig`.
- [x] 1.3 Add `src/common/configs/model.py` with passive `TrainingModelConfig`.
- [x] 1.4 Remove the old `src/common/components/training_components.py` module.
- [x] 1.5 Remove the old `src/data/components/config_models.py` runtime config module.

## 2. Python Import and Constructor Migration

- [x] 2.1 Update data code imports to use `src.common.configs.data`.
- [x] 2.2 Update quantization model imports, type annotations, defaults, and constructor parameter names to use `TrainingModelConfig` / `training_model_config`.
- [x] 2.3 Update TIGER model imports, type annotations, defaults, ignored hyperparameter name, and constructor parameter name to use `TrainingModelConfig` / `training_model_config`.
- [x] 2.4 Update focused tests to use `TrainingModelConfig` and `training_model_config`.

## 3. Hydra Config Migration

- [x] 3.1 Update all `configs/data/*.yaml` `_target_` values from `src.data.components.config_models.*` to `src.common.configs.data.*`.
- [x] 3.2 Update train model configs from `training_components` to `training_model_config`.
- [x] 3.3 Update train model config `_target_` values to `src.common.configs.model.TrainingModelConfig`.

## 4. OpenSpec Sync

- [x] 4.1 Update living data specs to require `src/common/configs/data.py` and `src.common.configs.data.*`.
- [x] 4.2 Update the unarchived `consolidate-model-training-components` artifacts to use `TrainingModelConfig` and `training_model_config`.

## 5. Verification

- [x] 5.1 Run residual scans for old class names, old module paths, and old config keys outside archive history.
- [x] 5.2 Run `openspec validate centralize-runtime-config-models --strict`.
- [x] 5.3 Run `openspec validate consolidate-model-training-components --strict`.
- [x] 5.4 Run focused `uv run ruff check` for touched code and tests.
- [x] 5.5 Run Hydra compose/instantiate smoke checks for affected data and model configs.
- [x] 5.6 Run focused tests covering updated direct-instantiation paths.
