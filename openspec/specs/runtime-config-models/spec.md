# runtime-config-models Specification

## Purpose
TBD - created by archiving change centralize-runtime-config-models. Update Purpose after archive.
## Requirements
### Requirement: Runtime config models SHALL live under common configs
Project-owned Hydra-instantiated runtime config dataclasses SHALL live under `src/common/configs/`, grouped by configuration domain.

#### Scenario: Data config models use the data domain module
- **WHEN** maintainers inspect dataset or dataloader config dataclasses
- **THEN** `DatasetConfig`, `SequenceDataloaderConfig`, and `ItemDataloaderConfig` MUST be defined in `src/common/configs/data.py`
- **THEN** those classes MUST NOT be defined in `src/data/components/config_models.py`

#### Scenario: Model training config uses the model domain module
- **WHEN** maintainers inspect model training dependency config classes
- **THEN** `TrainingModelConfig` MUST be defined in `src/common/configs/model.py`
- **THEN** the old `TrainingComponents` class MUST NOT remain as a runtime config class

### Requirement: Runtime config model targets SHALL use common configs paths
Official Hydra config files SHALL target project-owned runtime config model classes through `src.common.configs.*` paths.

#### Scenario: Data config targets use common data configs
- **WHEN** maintainers inspect `configs/data/*.yaml`
- **THEN** dataset and dataloader config `_target_` values MUST use `src.common.configs.data.*`
- **THEN** they MUST NOT use `src.data.components.config_models.*`

#### Scenario: Model training config targets use common model configs
- **WHEN** maintainers inspect train model configs
- **THEN** model training config `_target_` values MUST use `src.common.configs.model.TrainingModelConfig`
- **THEN** they MUST NOT use `src.common.components.training_components.TrainingComponents`

### Requirement: Common config models SHALL avoid domain implementation dependencies
Config classes under `src/common/configs/` SHALL NOT import implementation classes from higher-level domain packages such as `src.data`.

#### Scenario: Data config model typing avoids data package dependency
- **WHEN** maintainers inspect `src/common/configs/data.py`
- **THEN** it MUST NOT import from `src.data.*`
- **THEN** Hydra factory fields MAY use generic callable typing when a precise type would require a higher-level dependency

