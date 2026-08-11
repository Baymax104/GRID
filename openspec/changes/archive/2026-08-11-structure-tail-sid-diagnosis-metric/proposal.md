## Why

Tail-SID diagnosis currently concentrates metric logic in module-level functions, which makes the metric lifecycle and intermediate state hard to identify as the analysis grows. The diagnosis needs a clearer metric entrypoint without turning the offline analysis into a Lightning training or inference pipeline.

## What Changes

- Add a `TailSIDDiagnosisMetric` based on `torchmetrics.Metric` as the structured entrypoint for Tail-SID diagnosis metric computation.
- Move the current diagnosis metric computation behind the metric class and organize major calculation steps as methods on that class.
- Update the diagnosis runner flow to compute `DiagnosisResult` through `TailSIDDiagnosisMetric`.
- Let `TailSIDDiagnosisRunner` directly orchestrate input loading, frequency grouping, metric computation, and reporting without an intermediate `run_diagnosis(...)` function or diagnosis config wrapper.
- **BREAKING**: Remove the public `compute_metrics(...)` function and stop exporting it as the metric entrypoint.
- **BREAKING**: Remove the public `run_diagnosis(...)` helper and `DiagnosisConfig` wrapper from the Tail-SID diagnosis package.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `tail-sid-resolution-diagnosis`: Tail-SID diagnosis metric computation is structured behind a torchmetrics-compatible metric class, and the analysis runner directly owns diagnosis orchestration.

## Impact

- Affected code: `src/quantization/tail_sid_diagnosis/metrics.py`, `src/quantization/tail_sid_diagnosis/runner.py`, package exports, and Tail-SID diagnosis tests.
- API impact: callers must instantiate/update/compute `TailSIDDiagnosisMetric` instead of calling `compute_metrics(...)` directly; `run_diagnosis(...)` and `DiagnosisConfig` are no longer public package APIs.
- Dependency impact: no new dependency; `torchmetrics` is already part of the project runtime.
