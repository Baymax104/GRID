## ADDED Requirements

### Requirement: Official configs SHALL use data as the top-level data component name
Official experiment configs and launcher-facing configuration SHALL use `data` as the top-level component name for datamodule/dataloader/dataset related configuration instead of `data`.

#### Scenario: Launcher reads data entrypoint from top-level data group
- **WHEN** the launcher instantiates the datamodule for an official experiment
- **THEN** it MUST read the datamodule entrypoint from `data.datamodule`

### Requirement: Official experiment defaults SHALL mount split data configs under data
Official experiment entry files SHALL mount experiment-specific data config files under the top-level `data` key.

#### Scenario: Experiment defaults mount data config to data
- **WHEN** a maintainer inspects an official experiment defaults list
- **THEN** the experiment MUST mount its split data config under `@data`

### Requirement: Unused default component config stubs SHALL be removed
Default trainer/logger/callback config files that are no longer referenced by the official entrypoint chain SHALL not remain in the active config tree as dead stubs.

#### Scenario: Main config no longer depends on default trainer/logger/callback stubs
- **WHEN** a maintainer inspects `configs/main.yaml` and official experiment defaults
- **THEN** unused `trainer/default.yaml`, `logger/default.yaml`, and `callbacks/default.yaml` stubs MUST NOT remain as active dependencies

### Requirement: Support utilities SHALL align with the renamed data config key
Utility code that logs, warns on missing config sections, or prints the config tree SHALL use the renamed `data` key consistently.

#### Scenario: Config logging uses data key
- **WHEN** hyperparameters are logged or the config tree is printed
- **THEN** support utilities MUST refer to the top-level data config as `data`
