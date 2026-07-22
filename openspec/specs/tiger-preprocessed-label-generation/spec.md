# TIGER Preprocessed Label Generation

## Purpose

定义 TIGER sequence 训练样本在 preprocessing 阶段生成 `input_ids` 与 `target_ids` 的协议，使 `collate_fn_sequence` 只负责 batch 拼装。

## Requirements

### Requirement: TIGER label generation SHALL be a row-level preprocessing step
TIGER training and evaluation data preparation SHALL generate model `input_ids` and `target_ids` before `collate_fn_sequence` runs. Label generation MUST be declared as a preprocessing function in data configuration.

#### Scenario: preprocessing generates next item labels
- **WHEN** a TIGER preprocessing chain receives a semantic-ID-flattened sequence row
- **THEN** the label generation preprocessing step MUST produce `input_ids` containing the masked encoder input sequence
- **AND** it MUST produce `target_ids` containing the target semantic ID tensor for the next item
- **AND** `target_ids` MUST have shape `(num_hierarchies,)` for one row

#### Scenario: collate receives precomputed labels
- **WHEN** `collate_fn_sequence` receives rows after TIGER label preprocessing
- **THEN** each row MUST already contain the configured input field and target field
- **AND** `collate_fn_sequence` MUST NOT call label generator functions

### Requirement: TIGER SID causal duplicate SHALL be row-level preprocessing
TIGER training SID causal duplicate augmentation SHALL be expressed as a row expansion preprocessing step, not as a collate-time batch augmentation.

#### Scenario: training preprocessing expands sequence rows
- **WHEN** the training preprocessing chain applies SID causal duplicate expansion
- **THEN** the helper MUST yield contiguous semantic-ID-aligned subsequences
- **AND** each yielded subsequence MUST contain at least two items
- **AND** each yielded row MUST continue through downstream preprocessing steps such as label generation

#### Scenario: evaluation preprocessing is deterministic
- **WHEN** the validation or test preprocessing chain is configured
- **THEN** it MUST generate labels without applying random SID causal duplicate sampling
