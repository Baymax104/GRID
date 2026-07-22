## ADDED Requirements

### Requirement: TIGER train collate SHALL use a single public entry point
TIGER train/eval/test collate configuration SHALL use `collate_fn_train` as the single public collate entry point, with optional SID causal duplicate augmentation controlled by an explicit parameter.

#### Scenario: training collate enables augmentation through collate_fn_train
- **WHEN** 维护者检查 `configs/data/tiger_train.yaml` 的 `train_collate`
- **THEN** `_target_` MUST point to `src.data.components.collate.collate_fn_train`
- **AND** `enable_sid_causal_duplicate` MUST be set to `true`
- **AND** `sequence_field_name`、`sid_hierarchy`、`max_batch_size` MUST be declared in the train collate block

#### Scenario: eval and test collate use the same entry point without augmentation
- **WHEN** 维护者检查 `configs/data/tiger_train.yaml` 的 `eval_collate`
- **THEN** `_target_` MUST point to `src.data.components.collate.collate_fn_train`
- **AND** SID causal duplicate augmentation MUST be disabled or omitted so the default false value is used

#### Scenario: old wrapper collate entry point is removed
- **WHEN** 维护者检查 `src/data/components/collate.py`
- **THEN** it MUST NOT define `collate_with_sid_causal_duplicate`

### Requirement: SID causal duplicate sampling SHALL be isolated in a helper
SID causal duplicate sampling SHALL be implemented as a helper that only transforms a normalized dict batch and leaves padding, masking, label generation, and TIGER dataclass construction to `collate_fn_train`.

#### Scenario: helper preserves existing sampling semantics
- **WHEN** the helper receives a batch with a semantic ID sequence field
- **THEN** it MUST enumerate contiguous subsequences aligned by `sid_hierarchy`
- **AND** each selected subsequence MUST contain at least two items
- **AND** it MUST sample at most `max_batch_size` subsequences
- **AND** it MUST copy non-sequence fields from the original row for each selected subsequence

#### Scenario: augmentation parameters are required only when enabled
- **WHEN** `collate_fn_train` is called with `enable_sid_causal_duplicate=true`
- **THEN** it MUST require `sequence_field_name` and `sid_hierarchy`
- **AND** it MUST raise a clear error if either required parameter is missing
