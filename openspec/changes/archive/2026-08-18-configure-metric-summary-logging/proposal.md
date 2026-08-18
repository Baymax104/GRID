## Why

Tail-SID diagnosis produces one final set of analysis metrics, but the current metric runtime always routes logger-enabled metrics through Lightning history logging. W&B then presents these final scalar fields as charts, which is noisy for diagnosis runs and obscures that they are run-level values.

## What Changes

- Add a configurable metric logging mode to the shared metric callback for each stage.
- Keep the default mode as history logging, preserving existing train, validation, and test behavior.
- Add a summary-only mode for epoch-end metric stages that writes final scalar values to logger run summaries instead of metric history.
- Configure the Tail-SID diagnosis test metrics to use summary-only logging while preserving the existing `Trainer.test` pipeline and independent metric definitions.
- Add focused tests for history mode, summary mode, and diagnosis Hydra composition.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `metric-runtime-framework`: metric callbacks can choose history or summary output semantics for computed stage metrics.
- `tail-sid-resolution-diagnosis`: diagnosis test metrics use summary-only run-level logging for W&B-compatible scalar outputs.

## Impact

- Affected code: `src/common/metrics/callback.py`, `src/utils/launcher.py`, and focused metric callback tests.
- Affected config: `configs/model/tail_sid_diagnosis.yaml` or the metric callback attachment config surface used by `configs/experiment/tail_sid_diagnosis.yaml`.
- Affected systems: W&B logging behavior for diagnosis runs only; training experiments keep history curves by default.
- No new runtime dependency is required.
