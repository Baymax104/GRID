## MODIFIED Requirements

### Requirement: Tail SID diagnosis SHALL expose metric computation through torchmetrics
Tail-SID diagnosis metric computation SHALL be decomposed into focused `torchmetrics.Metric` classes for structural item metrics, semantic mismatch, damage scoring, and prefix risk. Shared diagnosis state SHALL be built once by the Lightning analysis module and passed to these metric components.

#### Scenario: Shared diagnosis context is built once
- **WHEN** the Tail-SID diagnosis LightningModule receives a test batch
- **THEN** it MUST build shared diagnosis context once from SID views, training frequencies, frequency groups, and optional embeddings
- **AND** metric components MUST reuse that context instead of independently rebuilding prefix buckets or group indexes

#### Scenario: Independent metrics preserve formulas
- **WHEN** the split metric components compute structural, semantic mismatch, damage, prefix, group, and summary outputs
- **THEN** item-level collision, density, suffix burden, semantic mismatch, damage, group rows, prefix rows, and summary fields MUST preserve their existing meanings
- **AND** reporting output file schemas MUST remain unchanged

#### Scenario: Lightning module assembles diagnosis result
- **WHEN** the Tail-SID diagnosis LightningModule completes `test_step`
- **THEN** it MUST assemble a `DiagnosisResult` from the split metric outputs
- **AND** it MUST log numeric summary fields through Lightning logging

#### Scenario: Old aggregate metric is removed
- **WHEN** a maintainer imports public Tail-SID diagnosis symbols
- **THEN** `TailSIDDiagnosisMetric` MUST NOT be exported
- **AND** diagnosis computation MUST use the shared context and split metric components
