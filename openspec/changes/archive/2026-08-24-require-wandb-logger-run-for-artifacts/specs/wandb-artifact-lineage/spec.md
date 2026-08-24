## ADDED Requirements

### Requirement: W&B artifact lifecycle SHALL be owned by WandbLogger

Official W&B-backed train, inference, and analysis experiments SHALL use the run exposed by the configured Lightning `WandbLogger.experiment` as the single W&B run used for config logging, metrics logging, Artifact publishing, and upstream Artifact lineage recording.

#### Scenario: Logger config is the run identity source
- **WHEN** maintainers inspect official W&B-backed experiment configs
- **THEN** W&B run identity fields MUST be declared under `configs/logger/*.yaml`
- **THEN** W&B writer callback configs MUST NOT duplicate run identity fields

#### Scenario: Writer does not manage run lifecycle
- **WHEN** maintainers inspect `WandbArtifactWriter` or `WandbCheckpointWriter`
- **THEN** those writers MUST NOT call `wandb.init`
- **THEN** those writers MUST NOT call `wandb.finish` or `run.finish`
- **THEN** those writers MUST publish only through the current logger-owned run

### Requirement: W&B artifact lineage callback SHALL record only on logger-owned runs

`WandbArtifactLineageCallback` SHALL record resolved upstream Artifact usage on the current logger-owned W&B run. It MUST NOT create, configure, finish, or replace a W&B run.

#### Scenario: Lineage records through logger-owned run
- **WHEN** a configured experiment resolves one or more `wandb://` input references
- **AND** `WandbArtifactLineageCallback` is enabled
- **AND** the trainer has a configured W&B logger whose `experiment` provides a run
- **THEN** the callback MUST call `use_artifact` on that logger-owned run for each resolved upstream Artifact

#### Scenario: Missing logger-owned run follows lineage failure policy
- **WHEN** resolved W&B artifact references exist
- **AND** `WandbArtifactLineageCallback` is enabled
- **AND** the trainer does not expose a configured W&B logger whose `experiment` provides a run
- **THEN** the callback MUST NOT call `wandb.init`
- **THEN** the callback MUST raise when `fail_on_missing_run` is true
- **THEN** the callback MUST warn and skip lineage recording when `fail_on_missing_run` is false

### Requirement: W&B writer and lineage behavior SHALL remain explicit

W&B logger configuration SHALL NOT automatically enable W&B artifact writers or lineage recording. Writers and lineage callbacks SHALL remain explicit callbacks, but when configured they MUST use the logger-owned run.

#### Scenario: Logger alone does not publish artifacts
- **WHEN** an experiment configures `WandbLogger`
- **AND** it does not configure a W&B writer callback
- **THEN** the experiment MUST NOT publish output Artifacts solely because the logger exists

#### Scenario: Callback presence requires logger-owned run
- **WHEN** an experiment configures a W&B writer callback or `WandbArtifactLineageCallback`
- **THEN** that callback MUST rely on the configured W&B logger-owned run
- **THEN** lifecycle ownership MUST remain with the logger and launcher finalization path
