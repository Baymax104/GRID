# python-component-entrypoint-normalization Specification

## Purpose
TBD - created by archiving. Update Purpose after archive.

## Requirements
### Requirement: Python-side top-level instantiation SHALL read from components entrypoints
The Python launcher layer SHALL obtain top-level instantiation roots for official experiments directly from `components` rather than through parameter-domain aliases.

#### Scenario: Datamodule and model roots are read from components
- **WHEN** the launcher instantiates an experiment datamodule or model
- **THEN** it MUST read the datamodule root from `components.data_loading.datamodule`
- **THEN** it MUST read the model root from `components.model.root`

#### Scenario: Trainer, callbacks, and logger roots are read from components
- **WHEN** the launcher instantiates trainer, callbacks, or loggers for an official experiment
- **THEN** it MUST read the trainer root from `components.trainer.root`
- **THEN** it MUST read callback definitions from `components.callbacks`
- **THEN** it MUST read logger definitions from `components.logger`

### Requirement: Parameter domains SHALL not act as Python instantiation aliases
Official experiment parameter domains such as `data_loading`, `model`, and `trainer` SHALL not retain top-level nodes whose only purpose is to forward Python direct-instantiation entrypoints to `components`.

#### Scenario: Parameter domain no longer forwards datamodule root
- **WHEN** a maintainer inspects an official experiment config
- **THEN** `data_loading` MUST NOT expose a top-level `datamodule` alias used only to forward the launcher to `components.data_loading.datamodule`

#### Scenario: Model parameter domain no longer serves as the Python root object
- **WHEN** a maintainer inspects an official experiment config
- **THEN** the top-level `model` domain MUST primarily express parameters and references
- **THEN** the Python-instantiated model root MUST live under `components.model.root`

### Requirement: Components entrypoint naming SHALL be consistent across official experiments
Official experiment configs SHALL use a consistent naming convention for Python-consumed component entrypoints.

#### Scenario: Entry naming matches the standard convention
- **WHEN** an official experiment defines component entrypoints for Python-side instantiation
- **THEN** it MUST use `components.data_loading.datamodule`
- **THEN** it MUST use `components.model.root`
- **THEN** it MUST use `components.trainer.root`
- **THEN** it MUST use `components.callbacks`
- **THEN** it MUST use `components.logger`
