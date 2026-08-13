## MODIFIED Requirements

### Requirement: Tail SID diagnosis SHALL compute structural damage metrics
The diagnosis SHALL compute item-level structural SID metrics from raw SID prefix buckets, including full collision, maximum prefix overlap depth, strict near-collision count, local density, suffix weakness, and last-step burden. The official analysis experiment SHALL compute these metrics through the shared metric runtime from `test_step` pre-state fields.

#### Scenario: Diagnosis structural metric is runtime-managed
- **WHEN** Tail-SID diagnosis runs as an analysis experiment
- **THEN** structural metric updates MUST be driven by `MetricEngine`
- **AND** `TailSIDDiagnosisModule.test_step` MUST NOT call `StructuralSIDMetric.update()` or `compute()` directly

### Requirement: Tail SID diagnosis SHALL compute semantic mismatch when embeddings are available
The diagnosis SHALL compute semantic mismatch for strict deep-overlap neighbors when `embedding_path` is provided. It SHALL skip semantic mismatch with zero-valued fields when embeddings are not provided. The official analysis experiment SHALL compute semantic mismatch through the shared metric runtime from `test_step` pre-state fields.

#### Scenario: Diagnosis semantic metric is runtime-managed
- **WHEN** Tail-SID diagnosis runs as an analysis experiment
- **THEN** semantic mismatch metric updates MUST be driven by `MetricEngine`
- **AND** `TailSIDDiagnosisModule.test_step` MUST NOT call `SemanticMismatchMetric.update()` or `compute()` directly

### Requirement: Tail SID diagnosis SHALL use stable damage normalization
Tail-SID diagnosis SHALL compute composite damage scores with a stable normalization strategy that does not amplify near-constant metric components into extremely large scores. The damage score metric SHALL be independent in the metric runtime and SHALL update from `test_step` pre-state fields rather than from another metric's computed output.

#### Scenario: Damage metric is independent
- **WHEN** the damage metric is updated by `MetricEngine`
- **THEN** it MUST receive only pre-state fields from the `test_step` payload
- **AND** it MUST NOT depend on `StructuralSIDMetric.compute()` or `SemanticMismatchMetric.compute()` outputs

### Requirement: Tail SID diagnosis SHALL emit reusable diagnosis outputs
The diagnosis SHALL create a clear output directory containing summary, group metrics, item damage scores, and prefix risk scores. The official analysis experiment SHALL assemble `DiagnosisResult` after metric runtime computation and before report writing.

#### Scenario: Diagnosis result is assembled after metric compute
- **WHEN** the Tail-SID diagnosis test epoch ends
- **THEN** a diagnosis result callback MUST compute test metrics through the metric engine
- **AND** it MUST assemble and store `pl_module.diagnosis_result`
- **AND** the report callback MUST write outputs from that stored result
