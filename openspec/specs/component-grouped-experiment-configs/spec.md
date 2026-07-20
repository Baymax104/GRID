# component-grouped-experiment-configs Specification

## Purpose
TBD - created by archiving change split-experiment-configs-by-component. Update Purpose after archive.
## Requirements
### Requirement: Official experiments SHALL be composed from per-component config groups
Each official experiment SHALL load its main component configurations from component-typed config groups rather than embedding the majority of component parameter trees directly inside `configs/experiment/<name>.yaml`.

#### Scenario: Experiment file imports component configs
- **WHEN** a maintainer inspects an official experiment entry file
- **THEN** that file MUST use defaults-based composition to load experiment-specific config files for component groups such as trainer, model, data_loading, logger, or callbacks

### Requirement: Experiment entry files SHALL remain thin assembly entrypoints
Official experiment entry files SHALL primarily express runtime metadata, manual inputs, and high-level composition/assembly concerns, instead of acting as the primary storage location for all component parameters.

#### Scenario: Main experiment file is shorter and assembly-focused
- **WHEN** a maintainer inspects an official experiment config after the refactor
- **THEN** the file MUST primarily contain experiment metadata, defaults imports, and necessary top-level assembly relations
- **THEN** the majority of trainer/model/data_loading/logger/callbacks parameter definitions MUST live in component-specific config files

### Requirement: Component configs SHALL be maintained once per experiment
For a given official experiment, component constructor parameters SHALL be maintained in the corresponding component config file instead of being mirrored through extensive parameter forwarding in the main experiment file.

#### Scenario: Trainer parameter change touches a single component file
- **WHEN** a maintainer adds or removes a trainer constructor parameter for one experiment
- **THEN** the change MUST primarily be made in that experiment's trainer config file
- **THEN** the main experiment file MUST NOT require a parallel field-by-field forwarding update for that parameter

### Requirement: Python-side components entrypoints SHALL remain explicit
Even after component configs are split out, the configuration assembled for Python SHALL continue to expose explicit `cfg.components` entrypoints for launcher consumption.

#### Scenario: Launcher still reads assembled components entrypoints
- **WHEN** the launcher instantiates datamodule, model, callbacks, logger, or trainer
- **THEN** the assembled config MUST continue to provide explicit component entrypoints under `cfg.components`

