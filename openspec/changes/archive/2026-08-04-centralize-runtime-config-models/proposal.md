## Why

Runtime config dataclasses are currently split by where they were first used: data configs live under `src/data/components/`, while model training config grouping lives under `src/common/components/`. These classes are Hydra-instantiated configuration models, so they should share one common namespace instead of being mixed with runtime components or domain implementation packages.

## What Changes

- Introduce `src/common/configs/` as the canonical package for Hydra-instantiated runtime config model classes.
- Move dataset and dataloader config dataclasses from `src/data/components/config_models.py` to `src/common/configs/data.py`.
- Rename the model training dependency config from `TrainingComponents` to `TrainingModelConfig` and move it to `src/common/configs/model.py`.
- Rename model config key and constructor parameter from `training_components` to `training_model_config`.
- Update Python imports, Hydra `_target_` strings, tests, and OpenSpec text to use the new package and class names.
- **BREAKING**: Old config class import paths and `training_components` config/constructor names are no longer supported.

## Capabilities

### New Capabilities

- `runtime-config-models`: Defines the canonical package and naming convention for Hydra-instantiated runtime config model classes.

### Modified Capabilities

- `data-config-class-convention`: Config class location changes from data-local to common config package.
- `data-model-role-separation`: Data config class references move out of `src/data/components/` while runtime batch data remains in data.
- `unified-dataset-config-contract`: Dataset and dataloader config `_target_` paths move to `src.common.configs.data`.

## Impact

- Affected code:
  - `src/common/configs/`
  - `src/data/components/config_models.py`
  - `src/data/datasets.py`
  - train-capable quantization and TIGER model constructors
- Affected config:
  - `configs/data/*.yaml`
  - `configs/model/*_train.yaml`
- Affected tests:
  - focused quantization constructor/config tests
  - Hydra compose/instantiate smoke checks
- Affected specs:
  - data config class location and target path requirements
  - unarchived model training config proposal artifacts
