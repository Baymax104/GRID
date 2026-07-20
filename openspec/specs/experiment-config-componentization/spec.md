# experiment-config-componentization Specification

## Purpose
TBD - created by archiving change componentize-experiment-configs. Update Purpose after archive.
## Requirements
### Requirement: Official experiment configs SHALL expose a components section for major assembly nodes
Each official experiment configuration SHALL define a top-level `components` section that holds the experiment's major instantiate-oriented configuration nodes, so the main execution chain is no longer buried inside parameter domains.

#### Scenario: Major data loading and model assembly nodes are named under components
- **WHEN** a maintainer opens an official experiment config
- **THEN** the config MUST expose a top-level `components` section
- **THEN** the main dataloader/dataset/model subtrees that define the experiment's primary assembly flow MUST appear as named nodes under `components`

### Requirement: Data loading and model domains SHALL primarily act as parameter domains
The `data_loading` and `model` top-level domains in official experiment configs SHALL primarily carry pure parameters, shared values, mappings, and references to named components rather than embedding most major instantiate subtrees inline.

#### Scenario: Parameter domains reference named components
- **WHEN** an official experiment config defines its datamodule or model assembly inputs
- **THEN** `data_loading` and `model` MUST primarily reference named nodes from `components`
- **THEN** they MUST NOT remain dominated by large inline instantiate subtrees for the experiment's primary assembly flow

### Requirement: Official experiments SHALL remain self-contained
Componentization SHALL happen within each experiment file; official experiment configs SHALL NOT depend on new repo-level shared component config files or cross-experiment component imports created solely for this restructuring.

#### Scenario: Experiment-local componentization
- **WHEN** an official experiment config is restructured
- **THEN** its componentized assembly nodes MUST remain defined within that experiment file
- **THEN** the restructuring MUST NOT require introducing repo-level shared component config files for reuse across experiments

### Requirement: Local inline targets MAY remain for minor one-off nodes
Componentization SHALL focus on major assembly nodes; small, strongly local, one-off `_target_` definitions MAY remain inline when extracting them would reduce readability.

#### Scenario: Minor one-off node remains inline
- **WHEN** a `_target_` node is short, local to one use site, and not a primary assembly node
- **THEN** the config MAY keep that node inline instead of lifting it into `components`

