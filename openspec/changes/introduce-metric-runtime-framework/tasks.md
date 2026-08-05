## 1. Core Runtime

- [x] 1.1 Create `src/common/metrics/` package exports.
- [x] 1.2 Implement metric input resolution for payload keys and indexed payload values.
- [x] 1.3 Implement `MetricEngine` with stage-isolated metrics, update, compute, reset, log, and repeat expansion.
- [x] 1.4 Implement `MetricCallback` for train, validation, and test batch/epoch hooks.

## 2. Tests

- [x] 2.1 Add unit tests for scalar metric update, compute, reset, and log output names.
- [x] 2.2 Add unit tests for repeat metric expansion and indexed payload values.
- [x] 2.3 Add unit tests for callback hook routing without running a full experiment.

## 3. Validation

- [x] 3.1 Run focused metric runtime tests.
- [x] 3.2 Run scoped ruff checks for the new metric package and tests.
