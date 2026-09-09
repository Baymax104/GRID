## MODIFIED Requirements

### Requirement: Official experiments SHALL own W&B project and group identity

Official experiment configs SHALL declare top-level `user`, `project`, and `group` fields for W&B logging, artifact publishing, and artifact lookup defaults. Component configs SHALL reference these top-level fields instead of hardcoding user/project/group values or declaring `wandb_project`. W&B group SHALL identify the quantization-method pipeline rather than the individual downstream task family.

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

#### Scenario: Fixed producer experiments use their pipeline group
- **WHEN** the experiment is RKMeans, RVQ, or RQ-VAE quantization train or inference
- **THEN** its group MUST respectively be `rkmeans`, `rvq`, or `rqvae`
- **AND** train and inference for the same quantization method MUST share that group

#### Scenario: Shared semantic embeddings use a public source group
- **WHEN** the experiment is `sem_embeds_inference`
- **THEN** its group MUST be `sem_embeds`

#### Scenario: Downstream experiments require quantization method group
- **WHEN** the experiment is `tiger_train`, `tiger_inference`, or `tail_sid_diagnosis`
- **THEN** its top-level group MUST be explicitly supplied as `rkmeans`, `rvq`, or `rqvae`
- **AND** it MUST NOT default to `tiger` or `tail_sid_diagnosis`

#### Scenario: Short W&B references use experiment user and project
- **WHEN** an experiment resolves a short W&B URI such as `wandb://<run-id>`
- **THEN** the default W&B entity MUST come from top-level `user`
- **THEN** the default W&B project MUST come from top-level `project`

## ADDED Requirements

### Requirement: Official W&B runs SHALL expose task and timestamp in the run name

Every official W&B logger configuration SHALL derive the run display name from the experiment task name and timezone-aware launch timestamp while preserving the existing W&B job type.

#### Scenario: Official logger composes run name
- **WHEN** an official experiment composes its W&B logger
- **THEN** `logger.wandb.name` MUST resolve as `${task_name}/${now_tz:%Y-%m-%d_%H-%M-%S}`
- **AND** `logger.wandb.job_type` MUST remain `train`, `inference`, or `analysis` according to the run mode

#### Scenario: Run name does not change local output identity
- **WHEN** the W&B run name uses the task/timestamp format
- **THEN** the experiment's Hydra `id` and output directory layout MUST remain unchanged
