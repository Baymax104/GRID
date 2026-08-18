## MODIFIED Requirements

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
