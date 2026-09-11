## ADDED Requirements

### Requirement: TIGER trace inference SHALL preserve target labels as diagnostic metadata
Trace inference SHALL use the unified sequence preprocessing and collate entrypoint while retaining generated `target_ids` in `TigerLabelData`. User identity SHALL remain output metadata only and MUST NOT become a generation feature.

#### Scenario: Trace batch is collated
- **WHEN** trace experiment preprocessing has produced `input_ids`, `attention_mask`, `target_ids`, and `user_id`
- **THEN** `collate_fn_sequence` MUST return `TigerModelInput` with user ID in `output_keys`
- **AND** it MUST return `TigerLabelData` containing `target_ids`
- **AND** user ID MUST NOT be included in encoder or decoder model inputs

### Requirement: TIGER trace inference SHALL declare its data split explicitly
The official trace experiment SHALL require a `data_split` value of `evaluation` or `testing` and SHALL construct its dataloader path from that value. The selected split MUST propagate to runtime config and trace Artifact metadata.

#### Scenario: Evaluation trace is configured
- **WHEN** `data_split=evaluation` is provided
- **THEN** the trace dataloader MUST read `${paths.data_dir}/evaluation`
- **AND** the resulting Artifact MUST record `evaluation`

#### Scenario: Testing trace is configured
- **WHEN** `data_split=testing` is provided after method freeze
- **THEN** the trace dataloader MUST read `${paths.data_dir}/testing`
- **AND** the resulting Artifact MUST record `testing`

#### Scenario: Trace split is missing or invalid
- **WHEN** an official trace experiment omits `data_split` or provides a value outside `evaluation|testing`
- **THEN** configuration or data initialization MUST fail before reading records
