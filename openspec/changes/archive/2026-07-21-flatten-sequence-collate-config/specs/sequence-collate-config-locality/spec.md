## ADDED Requirements

### Requirement: Sequence collate parameters SHALL be declared in collate config
Sequence dataloader collate parameters SHALL be declared on the configured collate callable rather than injected by the datamodule from sibling dataloader fields.

#### Scenario: Maintainer reads collate configuration
- **WHEN** a maintainer inspects a sequence experiment data YAML file
- **THEN** every non-batch argument passed to the collate function MUST be visible under the collate callable configuration used by that dataloader
- **AND** those arguments MUST NOT be hidden in `BaseDataModule._build_collate_fn()`

#### Scenario: BaseDataModule builds sequence collate function
- **WHEN** `BaseDataModule` builds a sequence dataloader
- **THEN** it MUST use the configured `collate_fn` as-is
- **AND** it MUST NOT wrap that callable with additional `labels`, `sequence_length`, `masking_token`, or `padding_token` bindings

### Requirement: Sequence dataloader config SHALL exclude collate-only fields
`SequenceDataloaderConfig` SHALL contain dataloader construction fields and MUST NOT expose fields used only to call sequence collate functions.

#### Scenario: SequenceDataloaderConfig fields are inspected
- **WHEN** maintainers inspect `SequenceDataloaderConfig`
- **THEN** it MUST NOT define `labels`, `sequence_length`, `masking_token`, or `padding_token`
- **AND** those values MUST be configured on the collate callable when required

#### Scenario: Item dataloader behavior remains unchanged
- **WHEN** item-level dataloaders build their collate function
- **THEN** they MUST continue to use the configured collate callable directly
- **AND** this change MUST NOT require item data configs to move existing collate parameters
