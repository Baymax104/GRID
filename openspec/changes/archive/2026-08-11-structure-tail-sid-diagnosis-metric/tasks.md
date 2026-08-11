## 1. Metric Structure

- [x] 1.1 Add `TailSIDDiagnosisMetric` as the public torchmetrics-based metric entrypoint.
- [x] 1.2 Move existing diagnosis metric computation into structured metric methods while preserving formulas and `DiagnosisResult`.

## 2. Integration

- [x] 2.1 Update diagnosis orchestration to use `TailSIDDiagnosisMetric` instead of `compute_metrics(...)`.
- [x] 2.2 Update package exports and tests to remove the public `compute_metrics(...)` API.
- [x] 2.3 Remove the intermediate `run_diagnosis(...)` helper and `DiagnosisConfig` wrapper from runner orchestration.

## 3. Verification

- [x] 3.1 Run focused Tail-SID diagnosis tests.
- [x] 3.2 Run focused linting on touched Python files.
