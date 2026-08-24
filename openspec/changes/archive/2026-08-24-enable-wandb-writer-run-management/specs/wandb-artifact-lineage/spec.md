## ADDED Requirements

### Requirement: W&B lineage callback SHALL NOT own run lifecycle
`WandbArtifactLineageCallback` SHALL remain responsible only for recording lineage against an active W&B run. It MUST NOT create W&B runs to compensate for writer run lifecycle.

> Superseded by `require-wandb-logger-run-for-artifacts`: lineage records only on the `WandbLogger`-owned run.

#### Scenario: Lineage callback does not initialize run
- **WHEN** resolved W&B input Artifact references exist
- **AND** there is no active W&B run
- **THEN** `WandbArtifactLineageCallback` MUST NOT call `wandb.init`
- **THEN** it MUST continue to skip or fail according to `fail_on_missing_run`

#### Scenario: Writer run management does not change lineage callback role
- **WHEN** W&B writers manage their own artifact publication run
- **THEN** `WandbArtifactLineageCallback` MUST remain a callback that reads the resolved-reference registry and calls `use_artifact` only when a run is already active
