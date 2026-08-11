## ADDED Requirements

### Requirement: Tail SID diagnosis SHALL expose metric computation through torchmetrics
Tail-SID diagnosis metric computation SHALL be exposed through a `TailSIDDiagnosisMetric` class that inherits from `torchmetrics.Metric`. The metric SHALL compute the same `DiagnosisResult` structure used by Tail-SID diagnosis reporting without requiring a Lightning trainer or model.

#### Scenario: Diagnosis result is computed through metric lifecycle
- **WHEN** a caller updates `TailSIDDiagnosisMetric` with SID views, training frequencies, frequency groups, and optional embeddings
- **THEN** `compute()` MUST return a `DiagnosisResult`
- **AND** the result MUST include summary, group rows, item rows, and prefix rows

#### Scenario: Metric remains offline-analysis compatible
- **WHEN** the Tail-SID analysis runner computes diagnosis metrics
- **THEN** it MUST instantiate and use `TailSIDDiagnosisMetric`
- **AND** it MUST NOT require `LightningModule`, `Trainer.predict`, or datamodule lifecycle setup
- **AND** it MUST NOT delegate metric orchestration through an intermediate `run_diagnosis(...)` helper or diagnosis config wrapper

### Requirement: Tail SID diagnosis SHALL remove the public compute_metrics entrypoint
Tail-SID diagnosis SHALL use `TailSIDDiagnosisMetric` as the public metric computation entrypoint. The module SHALL NOT expose `compute_metrics(...)` as a public function.

#### Scenario: Package exports structured metric entrypoint
- **WHEN** a maintainer imports public Tail-SID diagnosis symbols
- **THEN** `TailSIDDiagnosisMetric` MUST be exported
- **AND** `compute_metrics` MUST NOT be exported
- **AND** `run_diagnosis` MUST NOT be exported
- **AND** `DiagnosisConfig` MUST NOT be exported

#### Scenario: Existing formulas and output schemas are preserved
- **WHEN** diagnosis metrics are computed through `TailSIDDiagnosisMetric`
- **THEN** item-level collision, density, suffix burden, semantic mismatch, damage, group rows, prefix rows, and summary fields MUST preserve their existing meanings
- **AND** reporting output file schemas MUST remain unchanged
