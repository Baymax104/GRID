## MODIFIED Requirements

### Requirement: TIGER train collate SHALL use a single public entry point
TIGER train/eval/test collate configuration SHALL use `collate_fn_sequence` as the single public collate entry point. `collate_fn_sequence` SHALL be a pure batch assembly function over `list[dict[str, torch.Tensor]]` rows containing preprocessed fixed-length `input_ids`, `attention_mask`, and optional `target_ids`; SID causal duplicate augmentation, label generation, and sequence normalization MUST be declared as preprocessing rather than collate configuration.

#### Scenario: training collate uses collate_fn_sequence without augmentation or normalization controls
- **WHEN** 维护者检查 `configs/data/tiger_train.yaml` 的 `train_collate`
- **THEN** `_target_` MUST point to `src.data.components.collate.collate_fn_sequence`
- **AND** it MUST declare input, attention mask, and target field names
- **AND** it MUST NOT declare SID causal duplicate augmentation parameters
- **AND** it MUST NOT declare `sequence_length`

#### Scenario: eval and test collate use the same pure assembly entry point
- **WHEN** 维护者检查 `configs/data/tiger_train.yaml` 的 `eval_collate`
- **THEN** `_target_` MUST point to `src.data.components.collate.collate_fn_sequence`
- **AND** it MUST use the same input, attention mask, and target field contract as train collate
- **AND** it MUST NOT declare `sequence_length`

#### Scenario: old wrapper collate entry point is removed
- **WHEN** 维护者检查 `src/data/components/collate.py`
- **THEN** it MUST NOT define the old SID causal duplicate wrapper collate entry point

### Requirement: SID causal duplicate sampling SHALL be isolated in preprocessing
SID causal duplicate sampling SHALL be implemented as a row-level preprocessing helper that yields semantic-ID-aligned subsequence rows and leaves label generation and sequence normalization to downstream preprocessing steps, while leaving TIGER dataclass construction to `collate_fn_sequence`.

#### Scenario: helper preserves subsequence semantics
- **WHEN** the helper receives a row with a semantic ID sequence field
- **THEN** it MUST enumerate contiguous subsequences aligned by `sid_hierarchy`
- **AND** each selected subsequence MUST contain at least two items
- **AND** it MUST yield row dictionaries for downstream preprocessing

#### Scenario: collate no longer owns augmentation or normalization parameters
- **WHEN** `collate_fn_sequence` is called
- **THEN** it MUST NOT require `sequence_field_name` or `sid_hierarchy`
- **AND** it MUST NOT sample SID causal duplicate subsequences
- **AND** it MUST NOT require `sequence_length`
- **AND** it MUST NOT call `normalize_sequence_batch`
