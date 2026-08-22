# tiger-runtime-metrics Specification

## Purpose
TBD - created by archiving change migrate-tiger-metrics-to-runtime. Update Purpose after archive.
## Requirements
### Requirement: TIGER SHALL expose metric payloads instead of managing metrics
TIGER SHALL return metric payloads from train, validation, and test steps and SHALL NOT own framework-managed metric attributes or metric logging hooks.

#### Scenario: TIGER training step returns loss payload
- **WHEN** `training_step` runs with a valid TIGER training batch
- **THEN** it MUST return a mapping containing `loss`
- **THEN** it MUST NOT update or log framework-managed metrics directly

#### Scenario: TIGER validation and test steps return retrieval payload
- **WHEN** validation or test step runs with labels
- **THEN** it MUST return a mapping containing `loss`, `generated_ids`, `marginal_probs`, and `labels`
- **THEN** it MUST NOT reset or log framework-managed metrics directly

### Requirement: TIGER retrieval metrics SHALL use metric runtime
TIGER SID retrieval metrics SHALL be computed as concrete metric instances managed by `MetricEngine`.

#### Scenario: Retrieval metric adapter computes configured metric instances
- **WHEN** the metric runtime receives payload fields `marginal_probs`, `generated_ids`, and `labels`
- **THEN** it MUST use the configured SID retrieval metrics module to update each configured retrieval metric instance
- **THEN** compute MUST return metric names such as `ndcg@5` and `recall@10`

### Requirement: TIGER train config SHALL declare runtime metrics
The official TIGER train model config SHALL declare metrics through `model.metrics` and SHALL no longer pass `evaluator` to `model.root`.

#### Scenario: TIGER config declares loss and retrieval metrics
- **WHEN** a maintainer inspects `configs/model/tiger_train.yaml`
- **THEN** `model.metrics` MUST declare train loss, validation loss, test loss, validation retrieval, and test retrieval metrics
- **THEN** `model.root` MUST NOT pass an `evaluator` field

### Requirement: Pipeline SHALL attach metric callback from model metrics
The pipeline launcher SHALL attach the metric callback when model metrics are configured.

#### Scenario: Metric callback is attached for configured metrics
- **WHEN** pipeline modules are initialized with `cfg.model.metrics`
- **THEN** callbacks MUST include one metric callback using those metrics
- **THEN** models without `cfg.model.metrics` MUST keep their callback list unchanged

### Requirement: TIGER default training callbacks SHALL not early-stop on noisy retrieval metrics
The official TIGER training callback config SHALL keep early stopping disabled by default so validation retrieval metric jitter does not terminate standard step-budgeted training runs prematurely.

#### Scenario: Maintainer inspects TIGER train callbacks
- **WHEN** a maintainer inspects `configs/callbacks/tiger_train.yaml`
- **THEN** `early_stopping` MUST be `null`
- **AND** the default callback stack MUST NOT instantiate a Lightning `EarlyStopping` callback for `val/ndcg@10`
