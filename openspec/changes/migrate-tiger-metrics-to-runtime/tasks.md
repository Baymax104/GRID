## 1. Runtime Extensions

- [x] 1.1 Add `SIDRetrievalMetricGroup` under `src/common/metrics/`.
- [x] 1.2 Update metric package exports for the retrieval group.
- [x] 1.3 Add launcher auto-attachment for `MetricCallback` from `cfg.model.metrics`.

## 2. TIGER Migration

- [x] 2.1 Remove TIGER `evaluator` constructor dependency and metric attributes.
- [x] 2.2 Change TIGER training/validation/test steps to return metric payload mappings.
- [x] 2.3 Remove TIGER metric logging/reset hooks.
- [x] 2.4 Update `configs/model/tiger_train.yaml` to declare `model.metrics`.

## 3. Tests and Validation

- [x] 3.1 Add unit tests for `SIDRetrievalMetricGroup`.
- [x] 3.2 Add unit tests proving TIGER no longer exposes metric constructor arguments or metric attributes.
- [x] 3.3 Add unit tests for launcher metric callback attachment.
- [x] 3.4 Run focused tests without full experiments.
- [x] 3.5 Run scoped ruff checks.
