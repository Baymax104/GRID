## MODIFIED Requirements

### Requirement: TIGER train collate SHALL use a single public entry point
TIGER train/eval/test collate configuration SHALL use `collate_fn_train` as the single public collate entry point. `collate_fn_train` SHALL be a pure batch assembly function for preprocessed `input_ids` and `target_ids`; SID causal duplicate augmentation MUST be declared as preprocessing rather than collate configuration.

#### Scenario: training collate uses collate_fn_train without augmentation controls
- **WHEN** 维护者检查 `configs/data/tiger_train.yaml` 的 `train_collate`
- **THEN** `_target_` MUST point to `src.data.components.collate.collate_fn_train`
- **AND** it MUST declare input and target field names
- **AND** it MUST NOT declare SID causal duplicate augmentation parameters

#### Scenario: eval and test collate use the same pure assembly entry point
- **WHEN** 维护者检查 `configs/data/tiger_train.yaml` 的 `eval_collate`
- **THEN** `_target_` MUST point to `src.data.components.collate.collate_fn_train`
- **AND** it MUST use the same input and target field contract as train collate

#### Scenario: old wrapper collate entry point is removed
- **WHEN** 维护者检查 `src/data/components/collate.py`
- **THEN** it MUST NOT define the old SID causal duplicate wrapper collate entry point

### Requirement: SID causal duplicate sampling SHALL be isolated in preprocessing
SID causal duplicate sampling SHALL be implemented as a row-level preprocessing helper that yields semantic-ID-aligned subsequence rows and leaves padding, attention mask construction, and TIGER dataclass construction to `collate_fn_train`.

#### Scenario: helper preserves subsequence semantics
- **WHEN** the helper receives a row with a semantic ID sequence field
- **THEN** it MUST enumerate contiguous subsequences aligned by `sid_hierarchy`
- **AND** each selected subsequence MUST contain at least two items
- **AND** it MUST yield row dictionaries for downstream preprocessing

#### Scenario: collate no longer owns augmentation parameters
- **WHEN** `collate_fn_train` is called
- **THEN** it MUST NOT require `sequence_field_name` or `sid_hierarchy`
- **AND** it MUST NOT sample SID causal duplicate subsequences
