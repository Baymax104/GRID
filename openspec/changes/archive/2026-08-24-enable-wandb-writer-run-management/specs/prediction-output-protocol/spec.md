## REMOVED Requirements

### Requirement: W&B prediction artifact writer SHALL manage its own W&B run
**Reason**: Superseded by `require-wandb-logger-run-for-artifacts`; W&B writers now publish only through the `WandbLogger`-owned run.

**Migration**: Configure `logger.wandb` for W&B-backed inference and remove writer-level run lifecycle fields.

#### Scenario: Inference writer creates run when no logger exists
- **WHEN** inference enables `WandbArtifactWriter`
- **AND** `wandb.run` is not active
- **AND** the configured source file exists
- **THEN** this legacy behavior MUST NOT be used

#### Scenario: Inference writer reuses existing run
- **WHEN** inference enables `WandbArtifactWriter`
- **AND** a W&B logger or another owner has already created `wandb.run`
- **THEN** this legacy behavior MUST NOT be used

#### Scenario: Inference writer only finishes self-created run
- **WHEN** `WandbArtifactWriter` creates an artifact-only run itself
- **AND** `finish_run` is true
- **THEN** this legacy behavior MUST NOT be used
