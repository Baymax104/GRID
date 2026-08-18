## MODIFIED Requirements

### Requirement: Shared datamodule mechanics SHALL be centralized in the base class
The shared datamodule implementation SHALL centralize stage lifecycle mechanics separately from file-backed loading mechanics. A reusable stage-oriented base SHALL own stage configuration storage, setup-stage resolution, and dataloader hook dispatch. `FileDataModule` SHALL inherit that stage lifecycle base and retain stage readiness through its file map, file discovery, per-worker file assignment, dataset instantiation, collate selection, and common dataloader assembly for official file-backed loading flows.

#### Scenario: Common dataloader workflow is defined once
- **WHEN** a file-backed datamodule builds a train, validation, test, or predict dataloader
- **THEN** the common workflow for resolving stage config, binding files, constructing the dataset, and creating `DataloaderWithIterationRetry` MUST come from `FileDataModule`
- **AND** the common workflow for resolving setup stages and dispatching dataloader hooks MUST come from the shared stage lifecycle base

#### Scenario: Official datamodule target uses semantic file-backed name
- **WHEN** an official experiment configuration references a file-backed datamodule `_target_`
- **THEN** it MUST reference `src.data.datamodule.FileDataModule`
- **AND** it MUST NOT reference `src.data.data_module.BaseDataModule`

#### Scenario: Fit setup prepares validation files
- **WHEN** Lightning calls `setup("fit")` on `FileDataModule`
- **THEN** the datamodule MUST prepare both fitting and validating stages
- **AND** it MUST NOT prepare testing or predicting stages

#### Scenario: Evaluation setup prepares only the requested stage
- **WHEN** Lightning calls `setup("validate")`, `setup("test")`, or `setup("predict")` on `FileDataModule`
- **THEN** the datamodule MUST prepare only the corresponding validating, testing, or predicting stage
