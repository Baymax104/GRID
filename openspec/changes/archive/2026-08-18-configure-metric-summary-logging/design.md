## Context

`MetricCallback` currently owns update, logging, and reset hooks for metrics declared through `cfg.model.metrics`. Training logs on batch end, while validation and test metrics are computed at epoch end and passed to `LightningModule.log_dict(...)`. That keeps metric code decoupled from models, but it gives all logger-enabled metrics the same history-oriented semantics.

Tail-SID diagnosis is an analysis experiment that runs through `Trainer.test` and emits one final set of scalar metrics. When those scalars are sent to W&B history, the W&B UI treats them as chart fields even though they are run-level diagnosis results.

## Goals / Non-Goals

**Goals:**

- Add a metric callback logging mode that distinguishes history logging from run summary logging.
- Keep `MetricEngine` focused on update, compute, reset, and metric name construction.
- Preserve the current default behavior for training and existing metric users.
- Configure Tail-SID diagnosis test metrics to write W&B-compatible scalar summaries without writing metric history.
- Keep diagnosis on the unified `Trainer.test` path with independently configured metrics.

**Non-Goals:**

- Do not add a diagnosis-specific runner, callback, or aggregate metric.
- Do not change W&B logging behavior for training experiments.
- Do not introduce a new logging dependency or replace Lightning loggers.
- Do not reintroduce CSV logging for inference or diagnosis flows.

## Decisions

### 1. Put output semantics in `MetricCallback`

`MetricEngine` will continue returning computed metric mappings and calling `pl_module.log_dict(...)` for history logging. The choice between history and summary output belongs in `MetricCallback`, because it already owns stage hooks and has access to the `Trainer` and its loggers.

Alternative considered: add W&B-specific behavior to each diagnosis metric. This was rejected because metric classes should only compute values; logger backend behavior would leak into domain metrics.

### 2. Use explicit mode names: `history` and `summary`

The callback should accept stage-specific logging modes with `history` as the default and `summary` as the run-level mode. `history` keeps the existing `pl_module.log_dict(...)` path. `summary` computes prefixed metrics and writes them to logger run summaries.

Alternative considered: name the setting `curve` versus `scalar`. This is less precise because W&B curves are a UI representation of history data, while both modes still handle scalar metric values.

### 3. Support W&B summaries first, with safe fallback behavior

For W&B loggers, summary mode should write scalar-compatible values to `logger.experiment.summary`. For non-W&B loggers that do not expose a summary object, the callback should not fail the run; it should either skip summary logging with a rank-zero warning or use a clearly isolated fallback if the implementation can do so without creating history records.

Alternative considered: call `wandb.define_metric(..., hidden=True, summary="last")` while continuing to log history. This reduces chart noise but still writes metric history, so it does not satisfy strict summary-only semantics.

### 4. Configure diagnosis through the metric callback attachment surface

The launcher currently auto-attaches `MetricCallback(engine=...)` when `cfg.model.metrics` exists. The implementation should add a small config surface for callback logging options rather than requiring diagnosis to declare a custom callback. Tail-SID diagnosis can then set its test mode to `summary`; other experiments inherit defaults.

Alternative considered: move logging options into each metric definition. This would duplicate stage-level concerns across every metric and make consistent behavior harder to audit.

## Risks / Trade-offs

- [Risk] Lightning callback metrics may not include summary-only values if they bypass `pl_module.log_dict(...)`. -> Mitigation: summary mode should still return or store computed metrics in callback-owned state if downstream tests or logs need them, and tests should cover expected trainer-facing behavior where practical.
- [Risk] Logger APIs differ. -> Mitigation: implement summary writing by capability detection, with explicit support for W&B `experiment.summary` and a non-failing path for unsupported loggers.
- [Risk] Tensor values may not serialize cleanly into W&B summaries. -> Mitigation: convert detached scalar tensors to Python numeric values before summary assignment; reject or skip non-scalar values.
- [Risk] A too-broad config surface could complicate metric setup. -> Mitigation: keep modes stage-scoped and defaulted; diagnosis should be the only config override in this change.
