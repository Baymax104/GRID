## MODIFIED Requirements

### Requirement: Data, model, and analysis domains SHALL expose active component entrypoints
The `data` and `model` top-level domains in official Lightning-backed experiments SHALL expose the active Hydra `_target_` entrypoints consumed by Python launchers, while still referencing top-level manual inputs where appropriate. Official analysis experiments SHALL use the same Lightning component entrypoints instead of a separate `cfg.analysis.runner` entrypoint.

#### Scenario: Lightning experiment exposes active entrypoints
- **WHEN** an official train, inference, or analysis experiment config is composed
- **THEN** `cfg.data.datamodule` and `cfg.model.root` MUST identify the active data and model entrypoints

#### Scenario: Analysis experiment exposes test pipeline entrypoints
- **WHEN** an official analysis experiment config is composed
- **THEN** `cfg.data.datamodule` MUST identify the analysis datamodule
- **AND** `cfg.model.root` MUST identify the analysis LightningModule
- **AND** `cfg.trainer.root` MUST identify the analysis trainer

### Requirement: Official experiments MAY use repo-level component config files
Official experiment componentization SHALL use repo-level component config groups when those groups match the current repository structure.

#### Scenario: Experiment imports repo-level component config
- **WHEN** an official experiment needs data, model, trainer, logger, callbacks, or analysis configuration
- **THEN** it MAY import the corresponding config from `configs/<component>/<experiment>.yaml`
- **AND** it MUST keep manual input fields visible at the experiment entry level
