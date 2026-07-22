## MODIFIED Requirements

### Requirement: TIGER collate SHALL construct TIGER-specific batch objects
TIGER train and inference collate functions SHALL construct TIGER-specific model input and label data objects directly from the configured sequence field and output key field. TIGER train/eval/test SHALL use `collate_fn_train` as the single training collate entry point, with optional SID causal duplicate augmentation controlled by `enable_sid_causal_duplicate`.

#### Scenario: training collate returns model input and label data
- **WHEN** `collate_fn_train` receives rows containing the configured sequence field and label callable
- **THEN** it MUST return `(TigerModelInput, TigerLabelData)`
- **AND** `TigerModelInput.input_ids` MUST come from the label output masked input IDs
- **AND** `TigerModelInput.attention_mask` MUST be computed from `TigerModelInput.input_ids != padding_token`
- **AND** `TigerLabelData.target_ids` MUST come from the label output target IDs

#### Scenario: training collate optionally applies SID causal duplicate augmentation
- **WHEN** `collate_fn_train` is configured with `enable_sid_causal_duplicate=true`
- **THEN** it MUST apply SID causal duplicate sampling before padding and label generation
- **AND** it MUST still return `(TigerModelInput, TigerLabelData)`

#### Scenario: inference collate preserves output keys outside model input
- **WHEN** `collate_fn_inference_for_sequence` receives a field matching `id_field_name`
- **THEN** it MUST store that field in `TigerModelInput.output_keys`
- **AND** it MUST NOT store that field in `TigerModelInput.input_ids`
- **AND** attention masks MUST be computed from the non-id sequence input field
