# rootless-components-container-removal Specification

## Purpose
TBD - created by archiving change remove-components-config-layer. Update Purpose after archive.
## Requirements
### Requirement: Official configs SHALL expose top-level component entrypoints directly
Official experiment configurations SHALL expose Python-consumed component entrypoints directly at the top level instead of under a `components` wrapper.

#### Scenario: Launcher reads top-level component entrypoints
- **WHEN** the launcher instantiates official experiment components
- **THEN** it MUST read the datamodule from `data_loading.datamodule`
- **THEN** it MUST read the model from `model.root`
- **THEN** it MUST read the trainer from `trainer.root`
- **THEN** it MUST read callbacks from `callbacks`
- **THEN** it MUST read loggers from `logger`

### Requirement: Single-instance components SHALL retain root nodes
Single top-level components that also own nested helper subtrees SHALL retain a `root` node to identify the primary instantiated object.

#### Scenario: Model and trainer keep explicit root nodes
- **WHEN** a maintainer inspects split component config files for model or trainer
- **THEN** those configs MUST continue to expose their primary instantiated object under `root`

### Requirement: Component files SHALL not preserve redundant components wrapper structure
Per-component experiment config files SHALL not internally wrap their content in a redundant `components.<group>` layer when the file path already identifies the component group.

#### Scenario: Model file no longer nests under components.model
- **WHEN** a maintainer inspects a file such as `configs/model/<experiment>.yaml`
- **THEN** the file MUST define the `model` subtree content directly, without repeating an outer `model:` wrapper inside the file body
- **THEN** it MUST NOT additionally wrap the same content under `components.model`

### Requirement: Experiment defaults SHALL declare component mount points explicitly
Official experiment entry files SHALL use defaults package relocation to show where each split component config is mounted in the final config tree.

#### Scenario: Experiment defaults show mount targets
- **WHEN** a maintainer inspects an official experiment defaults list
- **THEN** the defaults entries MUST explicitly indicate mount points such as `@data_loading`, `@model`, `@trainer`, `@callbacks`, or `@logger`

### Requirement: Official configs SHALL not preserve duplicated parameter-domain views
After this refactor, official experiment configs SHALL not keep separate top-level parameter-only views whose values are mirrored into component entrypoints.

#### Scenario: Trainer parameter change does not require mirrored parameter domain update
- **WHEN** a maintainer adds or removes a trainer constructor parameter for one experiment
- **THEN** the change MUST be made in that experiment's trainer component config
- **THEN** it MUST NOT require maintaining a second top-level trainer parameter view

