## MODIFIED Requirements

### Requirement: Metric runtime SHALL manage metric lifecycle outside LightningModule
The system SHALL provide a reusable metric runtime that updates, computes, logs, and resets torchmetrics metrics from a Lightning callback instead of requiring each `LightningModule` to implement metric hooks. Analysis experiments SHALL be able to return a mapping of metric pre-state fields from `test_step` and rely on the metric runtime to update configured test metrics.

#### Scenario: Callback updates diagnosis metrics from test step payload
- **WHEN** an analysis `test_step` returns a mapping of metric pre-state fields
- **THEN** the metric callback MUST pass that payload to the metric engine
- **AND** the `LightningModule` MUST NOT update or compute framework-managed diagnosis metrics directly

### Requirement: Metric definitions SHALL be configuration driven
The metric runtime SHALL allow metrics to be instantiated from Hydra configuration so model configs can declare the metrics used by each model.

#### Scenario: Diagnosis metrics use explicit kwargs specs
- **WHEN** a diagnosis metric definition declares its update inputs
- **THEN** it MUST use explicit `spec.kwargs` entries for the required payload fields
- **AND** it MUST NOT require a pass-through adapter for fields already present in the step output mapping
