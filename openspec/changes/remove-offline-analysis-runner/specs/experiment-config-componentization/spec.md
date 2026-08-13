## MODIFIED Requirements

### Requirement: Data, model, and analysis domains SHALL expose active component entrypoints
The `data` and `model` top-level domains in official train, inference, and analysis experiment configs SHALL expose the active Hydra `_target_` entrypoints consumed by Python launchers. Official analysis experiments SHALL NOT expose `cfg.analysis.runner` as an official lifecycle entrypoint.

#### Scenario: Lightning experiment exposes active entrypoints
- **WHEN** an official train, inference, or analysis experiment config is composed
- **THEN** `cfg.data.datamodule` and `cfg.model.root` MUST identify the active data and model entrypoints

#### Scenario: Analysis experiment avoids runner entrypoint
- **WHEN** an official analysis experiment config is composed
- **THEN** it MUST NOT require `cfg.analysis.runner`
- **AND** it MUST provide Lightning `data`, `model`, `callbacks`, `logger`, and `trainer` component groups
