## Context

Tail-SID diagnosis now runs through Lightning `test` semantics and returns a metric pre-state mapping from `test_step`. The previous proposal introduced a diagnosis-specific metric callback to compute structured metric outputs, assemble `DiagnosisResult`, and hand it to a report callback. The updated architectural direction is stricter: metric recording must use the existing `MetricCallback`; artifact writing is not part of the current scope; diagnosis-specific callback classes should be deleted rather than layered around the framework.

## Goals / Non-Goals

**Goals:**

- Keep `TailSIDDiagnosisModule.test_step` as a pure pre-state producer.
- Use existing `MetricCallback` and `MetricEngine` without lifecycle extension or custom callback replacement.
- Make diagnosis metric outputs directly loggable by the existing `MetricCallback`.
- Temporarily remove official diagnosis artifact/report writing.
- Keep diagnosis computation logic inside metric classes.

**Non-Goals:**

- Do not reintroduce diagnosis output files in this change.
- Do not add a generic result-handler hook to `MetricCallback`.
- Do not extract shared diagnosis computation functions outside metric classes.
- Do not preserve `pl_module.diagnosis_result` as a writer handoff.

## Decisions

### Standard Metric Callback Only

The official diagnosis experiment will declare `model.metrics` and rely on `attach_metric_callback()` to attach `src.common.metrics.MetricCallback`. The previously added `model.metric_callback` override and `TailSIDDiagnosisMetricCallback` will be removed.

Alternative considered: keep a result handler hook on `MetricCallback`. Rejected because the user explicitly wants the current framework contract used as-is and prefers changing metric/test-step output shape over extending the framework.

### Independent Metric Output Shape

Diagnosis metrics configured in `MetricEngine` must return scalar values or scalar dictionaries that `MetricEngine.log()` can pass to Lightning `log_dict`. Structured intermediate values such as dataclasses, item rows, group rows, and prefix rows must not be configured as framework-logged metric outputs.

The official config will expand diagnosis into independent structural, semantic, damage, and prefix-risk metrics. Each metric stores the pre-state it needs, computes its own scalar dictionary, and does not depend on another configured metric's `compute()` output. Repeated internal computation is acceptable for this test-only analysis path.

### Artifact Writing Removed From Official Path

The diagnosis report callback will be removed from active config, and tests will no longer assert report files from the official experiment. Reporting helpers may be deleted if they become unused, or left only if still covered by non-runtime tests, but they must not be wired into the official diagnosis run.

## Risks / Trade-offs

- Loss of local report artifacts -> accepted temporarily because the current priority is architectural consistency around metric runtime.
- Repeated metric computation across independent metrics -> accepted because diagnosis is test-only and clarity of framework integration is more important than micro-optimization.
- Existing specs mention output files and offline runners -> this change explicitly removes or modifies those requirements to match the new direction.
