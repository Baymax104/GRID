## ADDED Requirements

### Requirement: W&B artifact downloads SHALL be rank-zero-only when distributed ranks share cache

When torch distributed is initialized, W&B artifact resolution SHALL download artifacts only on rank 0 and SHALL have other ranks resolve files from the same deterministic local download root after synchronization.

#### Scenario: Non-distributed download preserves current behavior
- **WHEN** W&B artifact resolution runs without initialized torch distributed
- **THEN** the resolver MUST call `artifact.download(...)`
- **THEN** the resolver MUST resolve the requested artifact file from the returned local directory

#### Scenario: Rank zero performs distributed download
- **WHEN** W&B artifact resolution runs with initialized torch distributed on rank 0
- **THEN** rank 0 MUST call `artifact.download(...)`
- **THEN** rank 0 MUST synchronize with other ranks before returning the resolved file path

#### Scenario: Non-zero rank waits for downloaded cache
- **WHEN** W&B artifact resolution runs with initialized torch distributed on a non-zero rank
- **THEN** the rank MUST NOT call `artifact.download(...)`
- **THEN** the rank MUST synchronize with rank 0
- **THEN** the rank MUST resolve the requested artifact file from the deterministic download root

#### Scenario: Missing shared cache is reported
- **WHEN** a non-zero rank cannot find the requested file under the deterministic download root after synchronization
- **THEN** the resolver MUST raise a file-not-found error that includes the requested file and local directory
