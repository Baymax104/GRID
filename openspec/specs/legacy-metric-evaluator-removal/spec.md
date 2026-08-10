# legacy-metric-evaluator-removal Specification

## Purpose
TBD - created by archiving change remove-legacy-metric-evaluators. Update Purpose after archive.
## Requirements
### Requirement: Legacy metric evaluator wrappers SHALL be removed
The project SHALL remove legacy evaluator wrapper classes after model metrics migrate to the runtime framework.

#### Scenario: Old evaluator classes are unavailable
- **WHEN** a maintainer inspects `src/common/components`
- **THEN** it MUST NOT contain `eval_metrics.py`

#### Scenario: TIGER retrieval metrics remain available
- **WHEN** runtime retrieval metrics are used
- **THEN** `NDCG` and `Recall` MUST remain importable from `src.recommendation.tiger.metrics`

