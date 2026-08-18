## ADDED Requirements

### Requirement: Official analysis experiments SHALL execute through Lightning test
Official analysis experiments SHALL use the shared Lightning pipeline assembly and SHALL execute through `Trainer.test(...)`.

#### Scenario: Analysis dispatch uses test lifecycle
- **WHEN** `src.main` receives a composed experiment with `run_mode: analysis`
- **THEN** it MUST instantiate the pipeline through `pipeline_launcher`
- **AND** it MUST call `trainer.test(...)`
- **AND** it MUST NOT call `trainer.fit(...)` or `trainer.predict(...)`

#### Scenario: Analysis pipeline uses standard components
- **WHEN** an official analysis experiment is composed
- **THEN** the config MUST provide `cfg.data.datamodule`
- **AND** it MUST provide `cfg.model.root`
- **AND** it MUST provide `cfg.callbacks`
- **AND** it MUST provide `cfg.logger`
- **AND** it MUST provide `cfg.trainer.root`

### Requirement: Analysis LightningModules SHALL be test-only
Analysis LightningModules SHALL compute analysis results through test lifecycle hooks and SHALL NOT expose a training optimizer requirement.

#### Scenario: Analysis module has no optimizer
- **WHEN** a Tail-SID diagnosis analysis module is instantiated
- **THEN** it MUST be usable by `Trainer.test(...)`
- **AND** it MUST NOT require `configure_optimizers()` to return an optimizer

#### Scenario: Numeric summary metrics are logged
- **WHEN** an analysis module computes a summary containing numeric and non-numeric fields
- **THEN** numeric fields MUST be logged through Lightning logging
- **AND** non-numeric fields MUST NOT be passed as scalar metrics

### Requirement: Analysis artifacts SHALL remain local and MAY be mirrored by logger callbacks
Analysis report files SHALL be written to the Hydra output directory as local artifacts, and callbacks MAY mirror them to the configured experiment logger.

#### Scenario: Local artifacts are written
- **WHEN** a Tail-SID diagnosis test run completes successfully
- **THEN** the Hydra output directory MUST contain the diagnosis JSON, CSV, and Markdown report files

#### Scenario: Logger-specific artifact upload is optional
- **WHEN** a callback runs with a logger that supports artifact upload
- **THEN** it MAY upload the local diagnosis files
- **AND** the diagnosis module MUST NOT import W&B directly
