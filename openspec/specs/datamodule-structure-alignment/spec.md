# datamodule-structure-alignment Specification

## Purpose
TBD - created by archiving change extract-datamodule-base-and-split-files. Update Purpose after archive.
## Requirements
### Requirement: DataModule implementation SHALL centralize shared file-loading concerns
The data loading layer SHALL define one reusable `BaseDataModule` for file-assignment, dataset instantiation, collate selection, and dataloader assembly behavior, and SHALL NOT expose separate sequence/item datamodule subclasses for the official pipelines.

#### Scenario: Official datamodule type is inspected
- **WHEN** developers inspect the datamodule type hierarchy
- **THEN** official data configs MUST target `BaseDataModule`
- **THEN** official code MUST NOT define `SequenceDataModule` or `ItemDataModule` subclasses

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

### Requirement: Task-specific collate behavior SHALL be configured on collate callables
Task-specific batching behavior SHALL live in the configured collate callable and its Hydra partial arguments, while `BaseDataModule` uses `curr_config.collate_fn` directly.

#### Scenario: Sequence dataloader uses configured collate callable
- **WHEN** `BaseDataModule` constructs a sequence dataloader
- **THEN** it MUST use the configured sequence collate callable directly
- **THEN** it MUST NOT bind sequence-only collate parameters from sibling dataloader fields

#### Scenario: Item dataloader uses configured collate callable
- **WHEN** `BaseDataModule` constructs an item dataloader
- **THEN** it MUST use the configured item collate callable directly

### Requirement: Datamodule module path SHALL reflect the unified implementation
The datamodule implementation SHALL live in `src/data/data_module.py` so the file path aligns with the flattened `src/data/` layout.

#### Scenario: Hydra targets use the new datamodule module layout
- **WHEN** an experiment configuration references a datamodule `_target_`
- **THEN** official experiments MUST reference `src.data.data_module.BaseDataModule`
- **THEN** configurations MUST NOT depend on `src.data.datamodules.*` or the legacy `src.data.loading.datamodules.*` layout

