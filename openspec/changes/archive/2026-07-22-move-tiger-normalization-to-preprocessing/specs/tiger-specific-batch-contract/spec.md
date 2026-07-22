## MODIFIED Requirements

### Requirement: TIGER collate SHALL construct TIGER-specific batch objects
TIGER train and inference SHALL use `collate_fn_sequence` to construct TIGER-specific model input and optional label data objects directly from configured input, attention mask, target, and output key fields on row dictionaries.

#### Scenario: training collate returns model input and label data
- **WHEN** `collate_fn_sequence` receives `list[dict[str, torch.Tensor]]` rows containing the configured input, attention mask, and target fields
- **THEN** it MUST return `(TigerModelInput, TigerLabelData)`
- **AND** `TigerModelInput.input_ids` MUST come from the preprocessed fixed-length input field
- **AND** `TigerModelInput.attention_mask` MUST come from the preprocessed attention mask field
- **AND** `TigerLabelData.target_ids` MUST come from the preprocessed target field

#### Scenario: training collate does not generate labels or normalize inputs
- **WHEN** `collate_fn_sequence` assembles a training batch
- **THEN** it MUST NOT call label generator functions
- **AND** it MUST NOT apply SID causal duplicate sampling
- **AND** it MUST NOT call `normalize_sequence_batch`
- **AND** it MUST NOT compute `attention_mask` from `padding_token`

#### Scenario: inference collate preserves output keys outside model input
- **WHEN** `collate_fn_sequence` receives a field matching `output_key_field_name`
- **THEN** it MUST store that field in `TigerModelInput.output_keys`
- **AND** it MUST NOT store that field in `TigerModelInput.input_ids`
- **AND** attention masks MUST come from the configured preprocessed attention mask field
