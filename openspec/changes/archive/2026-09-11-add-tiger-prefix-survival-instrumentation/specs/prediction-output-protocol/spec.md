## ADDED Requirements

### Requirement: Runtime prediction outputs MAY carry non-persistent auxiliary payloads
`ModelOutput` SHALL allow an optional named auxiliary payload for prediction callbacks while preserving `keys` and `predictions` as the only fields of the standard keyed prediction bundle. Existing callers that provide only keys and predictions MUST remain valid.

#### Scenario: Ordinary prediction output is created
- **WHEN** a model constructs `ModelOutput(keys, predictions)` without auxiliary data
- **THEN** existing writer and consumer behavior MUST remain unchanged

#### Scenario: Trace-enabled prediction output is created
- **WHEN** TIGER attaches a named Prefix Trace tensor payload
- **THEN** a dedicated auxiliary writer MUST be able to select that payload
- **AND** standard prediction writers MUST ignore it when producing `merged_predictions_tensor.pt`

### Requirement: Auxiliary tensor writers SHALL remain domain neutral
Shared auxiliary output writers SHALL operate on named tensor payloads and generic schema metadata without importing TIGER or Tail-SID diagnosis modules.

#### Scenario: Prefix trace payload is persisted
- **WHEN** a shared auxiliary writer receives the configured trace payload name
- **THEN** it MUST merge and persist the generic keyed tensor structure
- **AND** it MUST NOT branch on TIGER-specific metric names
