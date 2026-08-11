# analysis-runner-contract Specification

## Purpose
Define the common runtime contract for official offline analysis experiments.

## Requirements
### Requirement: Analysis runners SHALL execute through a common runner contract
Official analysis experiments SHALL instantiate an analysis runner from Hydra config and execute it through a common `run()` contract. Analysis runners SHALL NOT be required to subclass LightningModule, Callback, LightningDataModule, or Trainer.

#### Scenario: Runner is instantiated from config
- **WHEN** a `run_mode: analysis` experiment is launched
- **THEN** the system MUST instantiate the configured analysis runner from `cfg.analysis.runner`
- **AND** the instantiated runner MUST expose a callable `run()` method

#### Scenario: Runner executes without Lightning trainer
- **WHEN** the analysis runner executes
- **THEN** the analysis branch MUST call `runner.run()`
- **AND** it MUST NOT instantiate a Lightning Trainer solely to satisfy the analysis lifecycle

### Requirement: Analysis components SHALL live in common analysis and domain packages
Common analysis lifecycle code SHALL live under `src/common/analysis/`, while domain-specific analysis implementations SHALL live in their responsible domain package.

#### Scenario: Common analysis launcher placement
- **WHEN** a maintainer inspects the common analysis runner protocol or launcher
- **THEN** it MUST be located under `src/common/analysis/`
- **AND** it MUST NOT be located under `src/utils/`

#### Scenario: Quantization diagnosis runner placement
- **WHEN** a maintainer inspects Tail-SID diagnosis implementation
- **THEN** its domain-specific runner MUST remain under `src/quantization/tail_sid_diagnosis/`
- **AND** generic runner lifecycle code MUST NOT be embedded in the quantization diagnosis module

### Requirement: Analysis runners MAY reuse data-domain readers and datasets
Analysis runners MAY reuse `src.data` readers, datasets, dataloaders, and helper functions when they need data access. They SHALL NOT be forced to use Lightning `BaseDataModule` unless the analysis genuinely needs DataModule stage semantics.

#### Scenario: Runner uses data reader directly
- **WHEN** an analysis runner only needs to scan sequence records
- **THEN** it MAY use `src.data.components.readers` or `src.data.utils`
- **AND** it MUST NOT create a fake Lightning datamodule or trainer

#### Scenario: Runner uses dataloader when useful
- **WHEN** an analysis runner benefits from existing dataset or dataloader behavior
- **THEN** it MAY instantiate those data-domain components through Hydra
- **AND** it MUST keep analysis-specific orchestration outside `src/data`
