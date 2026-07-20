# datamodule-structure-alignment Specification

## Purpose
TBD - created by archiving change extract-datamodule-base-and-split-files. Update Purpose after archive.
## Requirements
### Requirement: Datamodule hierarchy SHALL separate shared file-loading concerns from task-specific batching behavior
The data loading layer SHALL define a neutral base datamodule for shared file-assignment and dataloader assembly behavior, and SHALL expose `SequenceDataModule` and `ItemDataModule` as sibling specializations rather than parent-child types.

#### Scenario: Sequence and item datamodules share a neutral parent
- **WHEN** developers inspect the datamodule type hierarchy
- **THEN** `SequenceDataModule` and `ItemDataModule` MUST both inherit from a shared base datamodule
- **THEN** `ItemDataModule` MUST NOT inherit from `SequenceDataModule`

### Requirement: Shared datamodule mechanics SHALL be centralized in the base class
The shared datamodule implementation SHALL centralize stage configuration storage, file discovery, per-worker file assignment, dataset instantiation, and common dataloader assembly in the base class so sibling datamodules do not duplicate the common loading flow.

#### Scenario: Common dataloader workflow is defined once
- **WHEN** a datamodule builds a train, validation, test, or predict dataloader
- **THEN** the common workflow for resolving stage config, binding files, constructing the dataset, and creating `DataloaderWithIterationRetry` MUST come from the shared base implementation

### Requirement: Task-specific collate behavior SHALL remain in the concrete datamodules
The concrete datamodules SHALL define their own batching-specific behavior so that sequence tasks can bind sequence-only collate parameters while item tasks can use their configured collate function directly.

#### Scenario: Sequence datamodule binds sequence-only collate parameters
- **WHEN** `SequenceDataModule` constructs a dataloader
- **THEN** it MUST provide a collate function that binds the configured sequence label and masking parameters before batch assembly

#### Scenario: Item datamodule keeps direct item collate behavior
- **WHEN** `ItemDataModule` constructs a dataloader
- **THEN** it MUST use the configured item collate function without inheriting sequence-only collate binding behavior

### Requirement: Datamodule module paths SHALL reflect separated roles
The datamodule package SHALL expose the base, sequence, and item implementations from separate module files so module paths align with the flattened `src/data/` layout.

#### Scenario: Hydra targets use the new datamodule module layout
- **WHEN** an experiment configuration references a datamodule `_target_`
- **THEN** sequence experiments MUST reference `src.data.datamodules.sequence.SequenceDataModule`
- **THEN** item-based experiments MUST reference `src.data.datamodules.item.ItemDataModule`
- **THEN** configurations MUST NOT depend on the old combined module path or the legacy `src.data.loading.datamodules.*` layout

