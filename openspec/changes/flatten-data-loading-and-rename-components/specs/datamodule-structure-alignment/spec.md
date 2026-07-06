## MODIFIED Requirements

### Requirement: Datamodule module paths SHALL reflect separated roles
The datamodule package SHALL expose the base, sequence, and item implementations from separate module files so module paths align with the flattened `src/data/` layout.

#### Scenario: Hydra targets use the new datamodule module layout
- **WHEN** an experiment configuration references a datamodule `_target_`
- **THEN** sequence experiments MUST reference `src.data.datamodules.sequence.SequenceDataModule`
- **THEN** item-based experiments MUST reference `src.data.datamodules.item.ItemDataModule`
- **THEN** configurations MUST NOT depend on the old combined module path or the legacy `src.data.loading.datamodules.*` layout
