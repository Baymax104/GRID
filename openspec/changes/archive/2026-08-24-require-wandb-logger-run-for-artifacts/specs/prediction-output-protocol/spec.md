## ADDED Requirements

### Requirement: W&B prediction artifact writer SHALL require a WandbLogger-owned run

`WandbArtifactWriter` SHALL publish inference output Artifacts only to the W&B run owned by the configured Lightning `WandbLogger`. It MUST NOT create, configure, finish, or otherwise manage a W&B run.

#### Scenario: Inference writer publishes through logger-owned run
- **WHEN** an inference experiment configures `WandbArtifactWriter`
- **AND** the trainer has a configured W&B logger whose `experiment` provides a run
- **THEN** `WandbArtifactWriter` MUST publish its merged model output bundle to that logger-owned run
- **THEN** the published Artifact metadata MUST include `role`, `task_name`, `local_output_path`, and `bundle_file`

#### Scenario: Missing logger-owned run fails
- **WHEN** `WandbArtifactWriter` reaches artifact publishing
- **AND** the trainer does not expose a configured W&B logger whose `experiment` provides a run
- **THEN** publishing MUST fail with a clear error naming the missing W&B logger run
- **THEN** `WandbArtifactWriter` MUST NOT call `wandb.init`

#### Scenario: Publish failure is not downgraded
- **WHEN** `WandbArtifactWriter` fails to create or log a W&B Artifact
- **THEN** the exception MUST propagate
- **THEN** `WandbArtifactWriter` MUST NOT provide a `fail_on_error` option
- **THEN** `WandbArtifactWriter` MUST NOT log a warning and continue as if W&B output succeeded

#### Scenario: Writer configuration excludes run lifecycle fields
- **WHEN** maintainers inspect official inference callback configs
- **THEN** `wandb_artifact_writer` MUST NOT declare `project`, `entity`, `group`, `run_name`, `job_type`, `tags`, `notes`, `mode`, `finish_run`, or `fail_on_error`
- **THEN** W&B run identity MUST be configured through the experiment's logger config
