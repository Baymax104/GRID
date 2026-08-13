## Why

Tail-SID diagnosis now runs through a Lightning test module, but the metric code still behaves as one large orchestration object. Splitting independent metric concerns lets `test_step` compute shared state once and compose focused metrics without duplicating prefix buckets, group indexes, or normalization context.

## What Changes

- Add a shared Tail-SID diagnosis context built once from the Lightning test batch.
- Split structural item metrics, semantic mismatch, damage scoring, prefix risk, and summary assembly into focused components.
- Update `TailSIDDiagnosisModule.test_step` to orchestrate context construction and independent metric computation.
- Preserve the existing `DiagnosisResult` output schema and report files.
- Keep DataModule, callback, W&B logger config, and local artifact writing behavior unchanged.

## Capabilities

### New Capabilities

### Modified Capabilities
- `tail-sid-resolution-diagnosis`: metric computation is decomposed into independent torchmetrics-compatible metric components while shared state is built once by the Lightning analysis module.

## Impact

- Affected code: `src/quantization/tail_sid_diagnosis/metrics.py`, `src/quantization/tail_sid_diagnosis/module.py`.
- Affected tests: Tail-SID metric and module tests.
- No config or dependency changes are expected.
