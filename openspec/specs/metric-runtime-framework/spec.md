# metric-runtime-framework Specification

## Purpose
TBD - created by archiving change introduce-metric-runtime-framework. Update Purpose after archive.
## Requirements
### Requirement: Metric runtime SHALL manage metric lifecycle outside LightningModule
The system SHALL provide a reusable metric runtime that updates, computes, logs, and resets torchmetrics metrics from a Lightning callback instead of requiring each `LightningModule` to implement metric hooks.

#### Scenario: Callback updates metrics from step output
- **WHEN** a train, validation, or test batch finishes and the model step returns a mapping output
- **THEN** the metric callback MUST pass the stage-specific payload to the metric engine
- **THEN** the `LightningModule` MUST NOT need to call metric update methods for framework-managed metrics

#### Scenario: Callback logs and resets epoch metrics
- **WHEN** a validation or test epoch ends
- **THEN** the metric callback MUST compute and log the stage metrics
- **THEN** the metric callback MUST reset that stage after logging

### Requirement: Metric definitions SHALL be configuration driven
The metric runtime SHALL allow metrics to be instantiated from Hydra configuration so model configs can declare the metrics used by each model.

#### Scenario: Stage metrics are instantiated from config
- **WHEN** a metric engine is constructed with train, validation, or test stage metric definitions
- **THEN** it MUST instantiate the configured torchmetrics metrics
- **THEN** it MUST keep stage metric state isolated

#### Scenario: Missing metrics do not break unrelated stages
- **WHEN** a stage has no configured metrics
- **THEN** the metric callback MUST skip updates and logging for that stage without failing

### Requirement: Metric runtime SHALL support dynamic metric expansion
The metric runtime SHALL support repeat definitions that expand one configured metric template into multiple concrete metrics.

#### Scenario: Repeat metrics expand by count
- **WHEN** a metric definition declares a repeat count and name template
- **THEN** the metric engine MUST create one metric instance per index
- **THEN** each expanded metric MUST use the rendered metric name

#### Scenario: Repeat metrics read indexed values
- **WHEN** an expanded metric definition declares a metric spec key and index placeholder
- **THEN** metric update MUST read the indexed value from the payload field
- **THEN** no dynamic `LightningModule` attribute is required for the expanded metric

### Requirement: Metric tests SHALL avoid full experiments
Metric framework verification SHALL use focused unit tests that construct minimal in-memory inputs and do not run full Hydra experiments, `src.main`, `torchrun`, or real Trainer training loops.

#### Scenario: Unit tests cover framework behavior
- **WHEN** metric runtime tests are executed
- **THEN** they MUST directly exercise engine, callback, and dynamic expansion behavior
- **THEN** they MUST NOT run a full experiment or require real data directories

