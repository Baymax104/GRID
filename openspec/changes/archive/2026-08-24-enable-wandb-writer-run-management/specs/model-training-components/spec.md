## REMOVED Requirements

### Requirement: W&B checkpoint writer SHALL manage its own W&B run
**Reason**: Superseded by `require-wandb-logger-run-for-artifacts`; checkpoint Artifact publishing now depends on the `WandbLogger`-owned run.

**Migration**: Configure `logger.wandb` for W&B-backed training and remove writer-level run lifecycle fields.

#### Scenario: Checkpoint writer creates run when no logger exists
- **WHEN** training enables `WandbCheckpointWriter`
- **AND** `wandb.run` is not active
- **AND** the selected checkpoint file exists
- **THEN** this legacy behavior MUST NOT be used

#### Scenario: Checkpoint writer reuses logger run
- **WHEN** training enables `WandbCheckpointWriter`
- **AND** W&B logger has already created `wandb.run`
- **THEN** this legacy behavior MUST NOT be used

#### Scenario: Checkpoint writer is independent from artifact writer
- **WHEN** maintainers inspect `src/common/writers/wandb_checkpoint_writer.py`
- **THEN** this legacy requirement MUST NOT be used as the current writer run lifecycle contract
