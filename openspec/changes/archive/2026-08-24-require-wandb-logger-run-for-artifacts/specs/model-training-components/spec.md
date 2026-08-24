## ADDED Requirements

### Requirement: W&B checkpoint writer SHALL require a WandbLogger-owned run

`WandbCheckpointWriter` SHALL publish checkpoint Artifacts only to the W&B run owned by the configured Lightning `WandbLogger`. It MUST NOT create, configure, finish, or otherwise manage a W&B run.

#### Scenario: Checkpoint writer publishes through logger-owned run
- **WHEN** a training experiment configures `WandbCheckpointWriter`
- **AND** the trainer has a configured W&B logger whose `experiment` provides a run
- **AND** the selected checkpoint file exists
- **THEN** `WandbCheckpointWriter` MUST publish the checkpoint Artifact to that logger-owned run
- **THEN** the published Artifact metadata MUST include `role: checkpoint`, `task_name`, `local_output_path`, and `bundle_file`

#### Scenario: Missing logger-owned run fails checkpoint publishing
- **WHEN** `WandbCheckpointWriter` reaches artifact publishing
- **AND** the trainer does not expose a configured W&B logger whose `experiment` provides a run
- **THEN** publishing MUST fail with a clear error naming the missing W&B logger run
- **THEN** `WandbCheckpointWriter` MUST NOT call `wandb.init`

#### Scenario: Checkpoint publish failure is not downgraded
- **WHEN** `WandbCheckpointWriter` fails to create or log a W&B Artifact
- **THEN** the exception MUST propagate
- **THEN** `WandbCheckpointWriter` MUST NOT provide a `fail_on_error` option
- **THEN** `WandbCheckpointWriter` MUST NOT log a warning and continue as if W&B output succeeded

#### Scenario: Checkpoint writer configuration excludes run lifecycle fields
- **WHEN** maintainers inspect official training callback configs
- **THEN** `wandb_checkpoint_writer` MUST NOT declare `project`, `entity`, `group`, `run_name`, `job_type`, `tags`, `notes`, `mode`, `finish_run`, or `fail_on_error`
- **THEN** W&B run identity MUST be configured through the experiment's logger config
