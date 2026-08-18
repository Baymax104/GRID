## MODIFIED Requirements

### Requirement: Tail SID diagnosis SHALL expose metric computation through torchmetrics
Tail-SID diagnosis metric computation SHALL be exposed through a `TailSIDDiagnosisMetric` class that inherits from `torchmetrics.Metric`. The metric SHALL compute the same `DiagnosisResult` structure used by Tail-SID diagnosis reporting and SHALL be usable inside a Lightning test-only analysis module.

#### Scenario: Diagnosis result is computed through metric lifecycle
- **WHEN** a caller updates `TailSIDDiagnosisMetric` with SID views, training frequencies, frequency groups, and optional embeddings
- **THEN** `compute()` MUST return a `DiagnosisResult`
- **AND** the result MUST include summary, group rows, item rows, and prefix rows

#### Scenario: Metric is usable by Lightning analysis
- **WHEN** the Tail-SID analysis LightningModule computes diagnosis metrics during test
- **THEN** it MUST instantiate and use `TailSIDDiagnosisMetric`
- **AND** it MUST preserve existing metric formulas and output schemas

### Requirement: Tail SID diagnosis SHALL run as an official analysis experiment
Tail-SID diagnosis SHALL be launched through the unified Hydra main entrypoint as a `run_mode: analysis` experiment. It SHALL run through the Lightning test lifecycle and SHALL NOT keep an independent argparse CLI as a parallel official entrypoint.

#### Scenario: Diagnosis experiment declares analysis mode
- **WHEN** a maintainer opens `configs/experiment/tail_sid_diagnosis.yaml`
- **THEN** the config MUST declare `run_mode: analysis`
- **AND** it MUST compose `data`, `model`, `callbacks`, `logger`, and `trainer` component configs for Tail-SID diagnosis

#### Scenario: Diagnosis script uses unified entrypoint
- **WHEN** a maintainer opens the root diagnosis shell script
- **THEN** it MUST call `uv run --module src.main experiment=tail_sid_diagnosis`
- **AND** it MUST NOT call `src.quantization.tail_sid_diagnosis.run` directly

#### Scenario: Independent CLI is removed
- **WHEN** a maintainer inspects `src/quantization/tail_sid_diagnosis/`
- **THEN** the package MUST NOT expose an argparse-based standalone CLI file for official diagnosis runs

### Requirement: Tail SID diagnosis SHALL write outputs through Lightning test lifecycle
Tail-SID diagnosis SHALL preserve existing local report outputs while triggering them from the Lightning test lifecycle.

#### Scenario: Diagnosis callback writes existing outputs
- **WHEN** Tail-SID diagnosis test completes successfully
- **THEN** the output directory MUST contain `summary.json`
- **AND** it MUST contain `group_metrics.csv`
- **AND** it MUST contain `item_damage_scores.csv`
- **AND** it MUST contain `prefix_risk_scores.csv`
- **AND** it MUST contain `report.md`

#### Scenario: Diagnosis summary metrics are logged
- **WHEN** Tail-SID diagnosis computes a result summary
- **THEN** numeric summary fields MUST be logged through the configured Lightning logger
- **AND** inference experiments MUST remain free of configured loggers by default
