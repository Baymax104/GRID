## ADDED Requirements

### Requirement: Prefix traces SHALL use an independent keyed tensor Artifact
Trace-enabled TIGER inference SHALL write `prefix_trace.pt` independently from the standard recommendation bundle. The file SHALL contain `schema_version`, `keys`, `labels`, `trace`, and `metadata`, and SHALL NOT change the contents of `merged_predictions_tensor.pt`.

#### Scenario: Prefix trace is written locally
- **WHEN** trace-enabled prediction completes with a local trace writer
- **THEN** `prefix_trace.pt` MUST be written under the configured Hydra output directory
- **AND** its keys, labels, and every batched trace tensor MUST have matching first-dimension lengths

#### Scenario: Standard recommendation output is written in the same run
- **WHEN** recommendation and trace writers consume the same prediction outputs
- **THEN** `merged_predictions_tensor.pt` MUST still contain only the existing `keys` and `predictions` bundle fields
- **AND** trace fields MUST exist only in `prefix_trace.pt`

### Requirement: Prefix Trace Artifact schema SHALL be stable and auditable
The trace schema SHALL use explicit tensor names, shapes, sentinels, version metadata, and source identity. Metadata MUST include data split, beam width, hierarchy count, codebook size, trace mode, checkpoint reference, and semantic-ID reference.

#### Scenario: Trace bundle is loaded
- **WHEN** a Prefix Trace Artifact is loaded for diagnosis
- **THEN** the loader MUST validate its schema version and required fields
- **AND** it MUST reject mismatched tensor lengths or hierarchy widths

#### Scenario: Missing ranks are represented
- **WHEN** a target prefix or parent is absent from a retained beam
- **THEN** the corresponding rank tensor MUST contain `-1`
- **AND** the survival tensor MUST contain false for that hierarchy

### Requirement: Prefix traces SHALL be publishable through the logger-owned W&B run
When a W&B logger is configured, the trace writer SHALL publish one Artifact with role/type `prefix_trace` through the active logger run. Local-only writing SHALL remain functional without initializing W&B.

#### Scenario: W&B trace publication succeeds
- **WHEN** trace-enabled inference completes with an active W&B logger
- **THEN** one Prefix Trace Artifact MUST contain `prefix_trace.pt`
- **AND** Artifact metadata MUST include schema version, data split, beam width, checkpoint and semantic-ID references

#### Scenario: No W&B logger is configured
- **WHEN** trace-enabled inference uses only a local writer
- **THEN** the local trace bundle MUST still be complete
- **AND** the writer MUST NOT initialize or finish a W&B run

### Requirement: Prefix trace writing SHALL be bounded and distributed-safe
The writer SHALL merge rank-local shards into one key-aligned bundle on rank zero and SHALL avoid retaining full `beam_width × codebook_size` candidate tensors in the output schema.

#### Scenario: Distributed prediction completes
- **WHEN** multiple prediction ranks emit trace shards
- **THEN** rank zero MUST merge shards by business key into one bundle
- **AND** duplicate keys MUST raise an error

#### Scenario: Trace payload is inspected
- **WHEN** maintainers inspect a saved trace bundle
- **THEN** it MUST contain target-centric statistics only
- **AND** it MUST NOT contain the full unpruned candidate cube for every layer
