## 1. Remove Custom Runtime Wiring

- [x] 1.1 Delete `TailSIDDiagnosisMetricCallback` and its tests.
- [x] 1.2 Remove `model.metric_callback` config and revert launcher custom metric-callback override logic.
- [x] 1.3 Ensure diagnosis metrics are attached through the standard `MetricCallback` from `model.metrics`.

## 2. Remove Official Artifact Writing

- [x] 2.1 Remove Tail-SID diagnosis report callback from official callback config.
- [x] 2.2 Remove `pl_module.diagnosis_result` handoff state and tests that depend on it.
- [x] 2.3 Delete or disconnect report/write-output tests that assert official diagnosis artifact generation.

## 3. Metric Shape

- [x] 3.1 Add or refactor diagnosis metric classes so official configured outputs are scalar values or scalar dictionaries.
- [x] 3.2 Keep structural, semantic, damage, and summary calculation logic inside metric classes without extracting shared pure functions.
- [x] 3.3 Update `configs/model/tail_sid_diagnosis.yaml` to configure separate loggable diagnosis metric outputs with explicit `spec.kwargs`.

## 4. Verification

- [x] 4.1 Update focused diagnosis, launcher, and metric framework tests.
- [x] 4.2 Run focused pytest and Ruff checks.
- [x] 4.3 Run residual scans for removed custom callback/report wiring.
- [x] 4.4 Validate `simplify-diagnosis-metric-runtime` with OpenSpec strict validation.
