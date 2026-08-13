## MODIFIED Requirements

### Requirement: Tail SID diagnosis SHALL compute structural damage metrics
The diagnosis SHALL compute structural SID information from raw SID prefix buckets as part of configured diagnosis metrics. The official analysis experiment SHALL expose loggable summary metrics through the shared metric runtime rather than writing item-level structural rows.

#### Scenario: Collision and near-collision summary metrics are recorded
- **WHEN** Tail-SID diagnosis runs as an analysis experiment
- **THEN** configured metric computation MUST account for full collisions and strict near-collisions
- **AND** the standard metric callback MUST record scalar summary values derived from those computations

#### Scenario: Prefix density summary metrics are recorded
- **WHEN** multiple items share raw SID prefixes
- **THEN** configured metric computation MUST account for prefix density
- **AND** the standard metric callback MUST record scalar summary values derived from density computations

#### Scenario: Suffix burden contributes to summary damage
- **WHEN** multiple items share the first `L-1` raw SID tokens without full collision
- **THEN** configured metric computation MUST account for last-step burden when producing scalar damage summary metrics

### Requirement: Tail SID diagnosis SHALL compute semantic mismatch when embeddings are available
The diagnosis SHALL compute semantic mismatch for strict deep-overlap neighbors when `embedding_path` is provided. It SHALL skip semantic mismatch with zero-valued contributions when embeddings are not provided. The official analysis experiment SHALL expose semantic mismatch through loggable summary metrics.

#### Scenario: Semantic mismatch uses item embeddings
- **WHEN** the diagnosis receives `embedding_path`
- **THEN** configured metric computation MUST compute cosine similarity for sampled strict deep-overlap neighbors
- **AND** the standard metric callback MUST record scalar semantic mismatch summary values

#### Scenario: Semantic mismatch is optional
- **WHEN** the diagnosis does not receive `embedding_path`
- **THEN** the diagnosis metric computation MUST still complete
- **AND** semantic mismatch contributions MUST be zero-valued in recorded summary metrics

### Requirement: Tail SID diagnosis SHALL use stable damage normalization
Tail-SID diagnosis SHALL compute composite damage scores with a stable normalization strategy that does not amplify near-constant metric components into extremely large scores. The official analysis experiment SHALL record scalar normalization and damage summary metrics through the shared metric runtime.

#### Scenario: Degenerate metric component is neutralized
- **WHEN** a metric component has an interquartile range below the configured stability threshold
- **THEN** that component MUST contribute zero to composite `damage`
- **AND** the system MUST NOT divide by a tiny epsilon in a way that produces unbounded scores

#### Scenario: Metric contribution is bounded
- **WHEN** a metric component has non-degenerate spread
- **THEN** its normalized contribution MUST be clamped to a finite configured range before being added to `damage`

#### Scenario: Score metadata is recorded as metrics
- **WHEN** diagnosis metrics are computed
- **THEN** scalar score normalization metadata MUST be available as logged metrics where numeric
- **AND** non-numeric explanatory metadata MUST NOT be sent to Lightning `log_dict`

### Requirement: Tail SID diagnosis SHALL run as an official analysis experiment
Tail-SID diagnosis SHALL be launched through the unified Hydra main entrypoint as a `run_mode: analysis` experiment using Lightning test semantics. It SHALL NOT keep an independent argparse CLI or offline analysis runner as a parallel official entrypoint.

#### Scenario: Diagnosis experiment declares analysis mode
- **WHEN** a maintainer opens `configs/experiment/tail_sid_diagnosis.yaml`
- **THEN** the config MUST declare `run_mode: analysis`
- **AND** it MUST compose Lightning data, model, trainer, logger, and metric configuration

#### Scenario: Diagnosis script uses unified entrypoint
- **WHEN** a maintainer opens the root diagnosis shell script
- **THEN** it MUST call `uv run --module src.main experiment=tail_sid_diagnosis`
- **AND** it MUST NOT call `src.quantization.tail_sid_diagnosis.run` directly

#### Scenario: Independent CLI is removed
- **WHEN** a maintainer inspects `src/quantization/tail_sid_diagnosis/`
- **THEN** the package MUST NOT expose an argparse-based standalone CLI file for official diagnosis runs

## REMOVED Requirements

### Requirement: Tail SID diagnosis SHALL emit reusable diagnosis outputs
**Reason**: Official diagnosis output writing is temporarily removed so the analysis experiment can first converge on the standard metric runtime contract.

**Migration**: Use logged scalar diagnosis metrics for current runs. Reintroduce file outputs later through a dedicated writer design that does not alter metric callback lifecycle.

### Requirement: Tail SID diagnosis SHALL emit a Markdown report
**Reason**: Markdown report generation depends on the temporarily removed artifact-writing path.

**Migration**: Use logged scalar diagnosis metrics for current runs. Reintroduce Markdown reporting later with the writer design.

### Requirement: Tail SID diagnosis SHALL show report location in stdout
**Reason**: The official experiment no longer writes `report.md`, so there is no report location to show.

**Migration**: Use logger output and metric dashboards for current diagnosis summaries.

### Requirement: Tail SID diagnosis SHALL print readable terminal tables
**Reason**: Terminal table rendering was tied to report/result artifact output and is outside the current metric-only diagnosis path.

**Migration**: Use logged scalar diagnosis metrics for current runs.

### Requirement: Tail SID diagnosis SHALL expose metric computation through torchmetrics
**Reason**: The previous structured `TailSIDDiagnosisMetric -> DiagnosisResult` contract conflicts with the current requirement that official metric framework outputs be directly loggable by the standard metric callback.

**Migration**: Use configured independent diagnosis metric classes whose `compute()` returns scalar values or scalar dictionaries.

### Requirement: Tail SID diagnosis SHALL use an analysis runner wrapper
**Reason**: The official diagnosis path now uses Lightning test semantics and the shared launcher, not a standalone analysis runner wrapper.

**Migration**: Use `experiment=tail_sid_diagnosis` through `src.main`, with data/model/trainer/logger/metric component configs.
