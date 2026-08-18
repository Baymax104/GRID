# metric-runtime-framework Specification

## Purpose
TBD - created by archiving change introduce-metric-runtime-framework. Update Purpose after archive.
## Requirements
### Requirement: Metric runtime SHALL manage metric lifecycle outside LightningModule
The system SHALL provide a reusable metric runtime that updates, computes, logs, and resets torchmetrics metrics from a Lightning callback instead of requiring each `LightningModule` to implement metric hooks. Analysis experiments SHALL use this existing callback lifecycle without custom metric callback replacements.

#### Scenario: Callback updates metrics from step output
- **WHEN** a train, validation, or test batch finishes and the model step returns a mapping output
- **THEN** the metric callback MUST pass the stage-specific payload to the metric engine
- **THEN** the `LightningModule` MUST NOT need to call metric update methods for framework-managed metrics

#### Scenario: Callback logs and resets epoch metrics
- **WHEN** a validation or test epoch ends
- **THEN** the metric callback MUST compute and log the stage metrics
- **THEN** the metric callback MUST reset that stage after logging

#### Scenario: Analysis metrics use standard callback lifecycle
- **WHEN** an analysis experiment declares metrics in model configuration
- **THEN** the launcher MUST attach the standard metric callback
- **AND** it MUST NOT require an analysis-specific metric callback class

### Requirement: Metric definitions SHALL be configuration driven
The metric runtime SHALL allow metrics to be instantiated from Hydra configuration so model configs can declare the metrics used by each model. Configured metric outputs that are logged by the standard callback SHALL be scalar values or scalar dictionaries accepted by Lightning logging.

#### Scenario: Stage metrics are instantiated from config
- **WHEN** a metric engine is constructed with train, validation, or test stage metric definitions
- **THEN** it MUST instantiate the configured torchmetrics metrics
- **THEN** it MUST keep stage metric state isolated

#### Scenario: Missing metrics do not break unrelated stages
- **WHEN** a stage has no configured metrics
- **THEN** the metric callback MUST skip updates and logging for that stage without failing

#### Scenario: Diagnosis metrics emit loggable values
- **WHEN** Tail-SID diagnosis metrics are configured for the test stage
- **THEN** their computed outputs MUST be directly loggable by the standard metric callback
- **AND** they MUST NOT rely on a custom callback to filter or transform structured outputs before logging

#### Scenario: Diagnosis metrics are independently configured
- **WHEN** Tail-SID diagnosis metrics are configured for the test stage
- **THEN** structural, semantic, damage, and prefix-risk metric concerns MUST be separate configured metrics
- **AND** the official config MUST NOT collapse them into one aggregate diagnosis metric

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

### Requirement: Metric callback SHALL support stage logging modes
The metric runtime SHALL allow each stage to choose whether computed metrics are emitted as logger history entries or as run summary values. The default mode SHALL be history logging to preserve existing behavior.

#### Scenario: History mode preserves existing logging
- **WHEN** a stage uses the default metric logging mode
- **THEN** `MetricCallback` MUST log computed metrics through the existing Lightning `log_dict` path
- **AND** existing `on_step`, `on_epoch`, `logger`, `prog_bar`, and `sync_dist` log kwargs MUST continue to apply

#### Scenario: Summary mode writes run-level values
- **WHEN** an epoch-end stage uses summary metric logging mode
- **THEN** `MetricCallback` MUST compute the prefixed stage metrics
- **AND** it MUST write scalar metric values to supported logger run summaries
- **AND** it MUST NOT write those metrics through the logger history path

#### Scenario: Unsupported summary logger does not fail metrics
- **WHEN** summary metric logging mode is active and a configured logger does not expose a supported summary destination
- **THEN** `MetricCallback` MUST NOT fail metric computation or stage reset
- **AND** it MUST make the unsupported summary logging behavior observable through a rank-zero warning or an equivalent testable signal

#### Scenario: Invalid logging mode fails clearly
- **WHEN** metric callback configuration declares an unsupported logging mode
- **THEN** callback construction MUST fail with a clear error naming the invalid mode

