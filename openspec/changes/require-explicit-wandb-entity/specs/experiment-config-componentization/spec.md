## MODIFIED Requirements

### Requirement: Official experiments SHALL own W&B user, project, and group identity

Official experiment configs SHALL declare top-level `user`, `project`, and `group` fields for W&B logging, artifact publishing, and artifact lookup defaults. Component configs SHALL reference these top-level fields instead of hardcoding user/project/group values or declaring top-level `wandb_project`.

#### Scenario: Experiment declares W&B identity
- **WHEN** a maintainer opens an official W&B-backed experiment config
- **THEN** the config MUST declare top-level `user`
- **THEN** the config MUST declare top-level `project`
- **THEN** the config MUST declare top-level `group`
- **THEN** the config MUST NOT declare top-level `wandb_project`

#### Scenario: Component configs reference experiment identity
- **WHEN** logger, callback, data, or model component configs need a W&B entity
- **THEN** they MUST reference `${user}` rather than hardcoding `baymaxam`, referencing environment variables, or omitting the entity
- **WHEN** logger, callback, data, or model component configs need a W&B project
- **THEN** they MUST reference `${project}` rather than hardcoding `GRID` or referencing `${wandb_project}`
- **AND** logger and writer component configs that need a W&B group MUST reference `${group}`

#### Scenario: Group uses experiment family
- **WHEN** an official experiment config declares `group`
- **THEN** the value MUST be the experiment family name rather than `${task_name}`
- **AND** train and inference experiments for the same family MUST share the same group value

#### Scenario: Short W&B references use experiment user and project
- **WHEN** an experiment resolves a short W&B URI such as `wandb://<run-id>`
- **THEN** the default W&B entity MUST come from top-level `user`
- **THEN** the default W&B project MUST come from top-level `project`
