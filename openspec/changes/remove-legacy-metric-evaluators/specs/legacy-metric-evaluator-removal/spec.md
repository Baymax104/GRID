## ADDED Requirements

### Requirement: Legacy metric evaluator wrappers SHALL be removed
The project SHALL remove legacy evaluator wrapper classes after model metrics migrate to the runtime framework.

#### Scenario: Old evaluator classes are unavailable
- **WHEN** a maintainer inspects `src/common/components/eval_metrics.py`
- **THEN** it MUST NOT define `Evaluator`
- **THEN** it MUST NOT define `SIDRetrievalEvaluator`

#### Scenario: Retrieval metrics remain available
- **WHEN** runtime retrieval metrics are used
- **THEN** `NDCG` and `Recall` MUST remain importable
