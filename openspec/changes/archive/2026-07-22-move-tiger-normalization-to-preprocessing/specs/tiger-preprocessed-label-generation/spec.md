## MODIFIED Requirements

### Requirement: TIGER label generation SHALL be a row-level preprocessing step
TIGER training and evaluation data preparation SHALL generate model `input_ids` and `target_ids` before `collate_fn_sequence` runs. Label generation MUST be declared as a preprocessing function in data configuration, and fixed-length input normalization MUST run after label generation and before `collate_fn_sequence`.

#### Scenario: preprocessing generates next item labels
- **WHEN** a TIGER preprocessing chain receives a semantic-ID-flattened sequence row
- **THEN** the label generation preprocessing step MUST produce `input_ids` containing the masked encoder input sequence
- **AND** it MUST produce `target_ids` containing the target semantic ID tensor for the next item
- **AND** `target_ids` MUST have shape `(num_hierarchies,)` for one row

#### Scenario: preprocessing normalizes model input after labels
- **WHEN** a TIGER preprocessing chain has generated `input_ids` and `target_ids`
- **THEN** the downstream `normalize_sequence` preprocessing step MUST produce fixed-length `input_ids`
- **AND** it MUST produce `attention_mask` aligned with the fixed-length `input_ids`
- **AND** it MUST preserve `target_ids`

#### Scenario: collate receives precomputed labels and normalized inputs
- **WHEN** `collate_fn_sequence` receives rows after TIGER label and normalization preprocessing
- **THEN** each row MUST already contain the configured input field, attention mask field, and target field
- **AND** `collate_fn_sequence` MUST NOT call label generator functions
- **AND** `collate_fn_sequence` MUST NOT normalize input sequence length

### Requirement: TIGER SID causal duplicate SHALL be row-level preprocessing
TIGER training SID causal duplicate augmentation SHALL be expressed as a row expansion preprocessing step, not as a collate-time batch augmentation.

#### Scenario: training preprocessing expands sequence rows
- **WHEN** the training preprocessing chain applies SID causal duplicate expansion
- **THEN** the helper MUST yield contiguous semantic-ID-aligned subsequences
- **AND** each yielded subsequence MUST contain at least two items
- **AND** each yielded row MUST continue through downstream preprocessing steps such as label generation and sequence normalization

#### Scenario: evaluation preprocessing is deterministic
- **WHEN** the validation or test preprocessing chain is configured
- **THEN** it MUST generate labels and normalize input sequences without applying random SID causal duplicate sampling
