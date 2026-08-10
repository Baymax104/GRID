# metric-input-adapters Specification

## Purpose
TBD - created by archiving change unify-metric-input-adapters. Update Purpose after archive.
## Requirements
### Requirement: Metric definitions SHALL use one metric spec resolution mechanism
The metric runtime SHALL update each configured metric instance through the metric definition `spec`.

#### Scenario: Scalar metric resolves payload key
- **WHEN** a metric definition declares `spec.key`
- **THEN** the metric engine MUST read that key from the stage payload
- **THEN** it MUST call the metric instance with the resolved value

#### Scenario: Keyword metric resolves payload kwargs
- **WHEN** a metric definition declares `spec.kwargs`
- **THEN** the metric engine MUST resolve each configured kwarg from the stage payload
- **THEN** it MUST call the metric instance with those keyword arguments

#### Scenario: Positional metric resolves payload args
- **WHEN** a metric definition declares `spec.args`
- **THEN** the metric engine MUST resolve each configured arg from the stage payload in order
- **THEN** it MUST call the metric instance with those positional arguments

### Requirement: Metric definitions SHALL support pure spec adapters
The metric runtime SHALL support an optional `spec.adapter` callable that converts the stage payload into keyword arguments for one metric instance.

#### Scenario: Adapter returns metric update kwargs
- **WHEN** a metric definition declares `spec.adapter`
- **THEN** the metric engine MUST call the adapter with the full stage payload
- **THEN** the adapter result MUST be used as keyword arguments for `metric.update`

#### Scenario: Adapter result must be a mapping
- **WHEN** an adapter returns a non-mapping value
- **THEN** the metric engine MUST raise a clear type error

#### Scenario: Adapter does not own metric lifecycle
- **WHEN** an adapter is used for a metric definition
- **THEN** the adapter MUST NOT compute, reset, log, or mutate the metric instance

### Requirement: Complex retrieval metrics SHALL be configured as concrete metric instances
Complex retrieval metrics SHALL be declared as independent metric definitions rather than through a metric group that expands or contains multiple metrics.

#### Scenario: TIGER retrieval metrics are flattened
- **WHEN** a maintainer inspects the official TIGER train model config
- **THEN** validation and test retrieval metrics MUST be declared as concrete metric entries such as `ndcg@5` and `recall@10`
- **THEN** each retrieval metric entry MUST configure its own `top_k`
- **THEN** each retrieval metric entry MUST use a spec adapter for SID retrieval payload conversion

#### Scenario: SID retrieval group is not used
- **WHEN** a maintainer scans source and official configs
- **THEN** they MUST NOT find `SIDRetrievalMetricGroup` as a runtime class, package export, or Hydra `_target_`

### Requirement: TIGER SID retrieval adapter SHALL preserve ranking semantics
The TIGER SID retrieval adapter SHALL convert TIGER evaluation payloads into the `preds`, `target`, and `indexes` inputs expected by retrieval metrics.

#### Scenario: Adapter converts generated SID candidates
- **WHEN** the adapter receives `marginal_probs`, `generated_ids`, and `labels`
- **THEN** it MUST use `marginal_probs` as flattened prediction scores
- **THEN** it MUST mark a candidate as target true only when the full generated SID equals the label SID
- **THEN** it MUST return flattened batch indexes aligned with the candidate dimension

