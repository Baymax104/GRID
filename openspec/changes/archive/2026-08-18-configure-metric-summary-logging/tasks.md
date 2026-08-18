## 1. Metric Callback Configuration

- [x] 1.1 Add a stage-scoped metric logging mode configuration surface for `MetricCallback`, defaulting every stage to `history`.
- [x] 1.2 Update launcher attachment so `cfg.model.metrics` can optionally provide callback logging options without requiring a custom callback declaration.
- [x] 1.3 Validate unsupported logging modes with a clear error during callback construction.

## 2. Summary Logging Implementation

- [x] 2.1 Keep `history` mode on the existing `MetricEngine.log(... pl_module.log_dict ...)` path.
- [x] 2.2 Implement `summary` mode for epoch-end validation and test metrics by computing prefixed stage metrics without sending them through logger history.
- [x] 2.3 Write scalar-compatible summary metrics to W&B-compatible `logger.experiment.summary` destinations.
- [x] 2.4 Convert scalar tensor metric values to Python scalars and skip or reject non-scalar values with observable rank-zero feedback.
- [x] 2.5 Ensure summary-mode stages still reset metric state after logging.

## 3. Tail-SID Diagnosis Configuration

- [x] 3.1 Configure `experiment=tail_sid_diagnosis` test metric logging mode as `summary`.
- [x] 3.2 Keep structural, semantic, damage, and prefix-risk metrics independently configured under the test stage.
- [x] 3.3 Keep diagnosis execution on the unified launcher and `Trainer.test` path.

## 4. Tests and Verification

- [x] 4.1 Add focused unit tests proving default history mode preserves existing `log_dict` behavior.
- [x] 4.2 Add focused unit tests proving summary mode writes final scalar values to a summary-capable logger without calling `log_dict`.
- [x] 4.3 Add focused unit tests for unsupported modes and unsupported summary logger behavior.
- [x] 4.4 Update diagnosis Hydra composition tests to assert test metric logging mode is `summary`.
- [x] 4.5 Run focused pytest for metric runtime, launcher metric callback attachment, and Tail-SID diagnosis tests.
- [x] 4.6 Run scoped Ruff on changed Python files and `openspec validate configure-metric-summary-logging --strict`.
