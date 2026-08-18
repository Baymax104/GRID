## Context

The diagnosis pipeline now uses Lightning `test` semantics and a shared diagnosis DataModule/Dataset. The remaining mismatch is metric lifecycle: `TailSIDDiagnosisModule.test_step` still constructs context, computes multiple metrics, assembles rows/summary, and logs summary values directly.

Existing `MetricEngine` expects:

```text
test_step output mapping -> MetricCallback -> MetricEngine.update(stage, payload)
```

This change aligns diagnosis with that contract.

## Design

### Test Step Payload

`TailSIDDiagnosisModule.test_step` SHALL build the metric pre-state explicitly and return a plain dictionary:

- `sid_views`
- `frequencies`
- `groups_by_item`
- `embeddings`
- `item_ids`
- `groups_by_index`
- `buckets`
- `strict_depth`
- `sid_length`

The module SHALL NOT update or compute diagnosis metrics in `test_step`.

### Independent Metrics

All diagnosis metrics SHALL update from raw pre-state fields, not from another metric's `compute()` output:

- `StructuralSIDMetric`
- `SemanticMismatchMetric`
- `DamageScoreMetric`
- `PrefixRiskMetric`

`DamageScoreMetric` and `PrefixRiskMetric` may recompute structural or semantic component values internally to remain independent. This is acceptable because diagnosis is a test-only full-batch analysis path and clarity is more important than avoiding repeated computation.

### Metric Configuration

The `MetricEngine` test-stage config SHALL use explicit `spec.kwargs`.

No adapter SHALL be introduced for pass-through diagnosis payload fields.

### Result Assembly

Add a diagnosis result callback that:

- reads the `MetricEngine` attached to metric callback
- computes test metric outputs
- assembles `DiagnosisResult` from metric outputs and the latest test payload state
- stores `pl_module.diagnosis_result`
- logs numeric summary values

The existing report callback remains responsible for writing files and saving artifacts with the logger.

## Non-Goals

- Do not introduce a general metric dependency graph.
- Do not add pass-through metric adapters.
- Do not change report output schemas or metric formulas.
