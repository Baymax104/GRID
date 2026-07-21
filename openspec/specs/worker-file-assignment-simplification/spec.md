# worker-file-assignment-simplification Specification

## Purpose
TBD - created by archiving change remove-assign-all-files-per-worker. Update Purpose after archive.
## Requirements
### Requirement: Official dataloader configs SHALL not expose assign-all-files worker mode
The official data loading configuration surface SHALL not expose a mode where every dataloader worker receives the full file set for a stage.

#### Scenario: Official experiments omit the removed config
- **WHEN** a maintainer inspects official experiment dataloader configuration
- **THEN** the configuration MUST NOT declare `assign_all_files_per_worker`

### Requirement: Worker file assignment SHALL use standard per-worker file partitioning
The loading pipeline SHALL assign files to workers using the standard partitioned file-distribution flow rather than a special all-files-per-worker branch.

#### Scenario: Dataset worker file selection uses partitioned file lists
- **WHEN** a dataset worker resolves the files it should process
- **THEN** it MUST derive its file list from the worker-partitioned stage assignment
- **THEN** it MUST NOT rely on an official configuration mode that bypasses file partitioning by duplicating the entire file list to each worker

### Requirement: Datamodule behavior SHALL not depend on assign-all-files validation
The datamodule layer SHALL not require stage-specific validation or branching tied to the removed assign-all-files worker mode.

#### Scenario: Unified datamodule no longer guards the removed mode
- **WHEN** `BaseDataModule` constructs a dataloader
- **THEN** it MUST NOT perform stage validation that exists only to police the removed `assign_all_files_per_worker` capability
