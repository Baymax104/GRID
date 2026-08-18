## Why

Tail-SID diagnosis has drifted toward extra callback seams for result assembly and report writing, while the current target is simpler: analysis `test_step` returns metric pre-state and the existing metric framework records all diagnosis metrics. Diagnosis should not extend the metric callback lifecycle or keep temporary writer/report wiring while the metric shape is still being settled.

## What Changes

- Remove the diagnosis-specific metric callback path and rely on the standard `MetricCallback` attached from `model.metrics`.
- Temporarily remove Tail-SID diagnosis artifact/report writing from the official experiment path.
- Remove `pl_module.diagnosis_result` as an intermediate handoff state.
- Change diagnosis runtime metrics so configured `MetricEngine` outputs are loggable scalar summary values.
- Keep diagnosis computation logic inside metric classes; do not extract shared pure computation functions during this change.
- **BREAKING**: The official Tail-SID diagnosis experiment will no longer write `summary.json`, CSV outputs, or `report.md` until the writer design is reintroduced explicitly.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `metric-runtime-framework`: clarify that diagnosis must use the existing metric callback without custom callback extensions and must emit loggable metric outputs.
- `tail-sid-resolution-diagnosis`: change the official diagnosis experiment to record summary metrics only and temporarily remove artifact/report output requirements.

## Impact

- Affected code: `src/quantization/tail_sid_diagnosis/module.py`, `metrics.py`, `callbacks.py`, configs under `configs/model` and `configs/callbacks`, launcher changes introduced for custom metric callbacks, and focused tests.
- Public behavior impact: official diagnosis runs log W&B/scalar metrics but do not write local diagnosis report artifacts.
- Dependency impact: none.
