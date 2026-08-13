## Why

Tail-SID diagnosis currently computes and assembles metrics directly inside `test_step`, which bypasses the shared metric runtime used by other experiments. Moving diagnosis metrics into `MetricEngine` makes analysis experiments follow the same lifecycle: step returns metric pre-state, callback updates configured metrics, and callbacks handle result assembly/reporting.

## What Changes

- Make `TailSIDDiagnosisModule.test_step` return an expanded metric pre-state dictionary instead of computing metrics.
- Configure all Tail-SID diagnosis metrics as independent `MetricEngine` test-stage metrics.
- Do not use metric adapters; each metric declares explicit `spec.kwargs` fields.
- Refactor damage and prefix-risk metrics so they update from the raw pre-state fields and do not depend on other metric `compute()` outputs.
- Add a diagnosis result callback that computes metric outputs from the engine, assembles `DiagnosisResult`, stores it on the module, and logs numeric summary values.
- Keep report writing in the existing report callback.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `metric-runtime-framework`: support structured diagnosis metrics through existing mapping payload and explicit kwargs specs.
- `tail-sid-resolution-diagnosis`: require the official diagnosis experiment to compute metrics through the shared metric runtime.

## Impact

- Affected code: `src/quantization/tail_sid_diagnosis/module.py`, `metrics.py`, callbacks/config/tests, and `configs/model/tail_sid_diagnosis.yaml`.
- Public config impact: Tail-SID diagnosis model config gains a `metrics: MetricEngine` section.
- Dependency impact: none.
