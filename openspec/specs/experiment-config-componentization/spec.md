# experiment-config-componentization Specification

## Purpose
TBD - created by archiving change componentize-experiment-configs. Update Purpose after archive.
## Requirements
### Requirement: Official experiment configs SHALL use component config groups for major assembly nodes
Each official experiment configuration SHALL compose major instantiate-oriented configuration nodes from repo-level component config groups, so the execution chain is not buried inside parameter domains.

#### Scenario: Major assembly nodes are provided by component groups
- **WHEN** a maintainer opens an official experiment config
- **THEN** the config MUST use defaults entries such as `/data@data`, `/model@model`, `/trainer@trainer`, `/callbacks@callbacks`, `/logger@logger`, or `/analysis@analysis` as applicable
- **THEN** the primary assembly flow MUST NOT depend on an obsolete experiment-local `components` section

### Requirement: Data, model, and analysis domains SHALL expose active component entrypoints
The `data` and `model` top-level domains in official train, inference, and analysis experiment configs SHALL expose the active Hydra `_target_` entrypoints consumed by Python launchers. Official analysis experiments SHALL NOT expose `cfg.analysis.runner` as an official lifecycle entrypoint.

#### Scenario: Lightning experiment exposes active entrypoints
- **WHEN** an official train, inference, or analysis experiment config is composed
- **THEN** `cfg.data.datamodule` and `cfg.model.root` MUST identify the active data and model entrypoints

#### Scenario: Analysis experiment avoids runner entrypoint
- **WHEN** an official analysis experiment config is composed
- **THEN** it MUST NOT require `cfg.analysis.runner`
- **AND** it MUST provide Lightning `data`, `model`, `callbacks`, `logger`, and `trainer` component groups

### Requirement: Official experiments MAY use repo-level component config files
Official experiment componentization SHALL use repo-level component config groups when those groups match the current repository structure.

#### Scenario: Experiment imports repo-level component config
- **WHEN** an official experiment needs data, model, trainer, logger, callbacks, or analysis configuration
- **THEN** it MAY import the corresponding config from `configs/<component>/<experiment>.yaml`
- **AND** it MUST keep manual input fields visible at the experiment entry level

### Requirement: Local inline targets MAY remain for minor one-off nodes
Componentization SHALL focus on major assembly nodes; small, strongly local, one-off `_target_` definitions MAY remain inline when extracting them would reduce readability.

#### Scenario: Minor one-off node remains inline
- **WHEN** a `_target_` node is short, local to one use site, and not a primary assembly node
- **THEN** the config MAY keep that node inline instead of lifting it into a component config group
