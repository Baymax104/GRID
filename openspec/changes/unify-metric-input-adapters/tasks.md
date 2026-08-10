## 1. Metric Runtime

- [x] 1.1 Extend `MetricEngine` spec resolution so `spec.adapter` instantiates/calls a callable with the full payload and passes the returned mapping to `metric.update(**kwargs)`.
- [x] 1.2 Normalize existing scalar input configs to the same `spec` mechanism while preserving string shorthand compatibility.
- [x] 1.3 Add clear type errors for adapter results that are not mappings.
- [x] 1.4 Remove the `update_from_payload` runtime branch after confirming no remaining call sites need it.

## 2. TIGER Adapter

- [x] 2.1 Add `src/recommendation/tiger/metric_adapters.py` with a pure `sid_retrieval_inputs(payload)` adapter.
- [x] 2.2 Unit-test the adapter with minimal `marginal_probs`, `generated_ids`, and `labels` tensors.
- [x] 2.3 Ensure the adapter returns `preds`, `target`, and `indexes` on the correct device and with candidate-aligned flattened shapes.

## 3. Config and Cleanup

- [x] 3.1 Flatten `configs/model/tiger_train.yaml` retrieval metrics into concrete `ndcg@5`, `ndcg@10`, `recall@5`, and `recall@10` metric entries using `spec.adapter`.
- [x] 3.2 Delete `SIDRetrievalMetricGroup`, remove it from `src.common.metrics` exports, and remove old group tests.
- [x] 3.3 Add config tests proving TIGER retrieval metrics no longer use `SIDRetrievalMetricGroup` and each metric owns its own `top_k`.
- [x] 3.4 Run residual scans for `SIDRetrievalMetricGroup`, `update_from_payload`, and old retrieval group config shapes.

## 4. Artifacts and Validation

- [x] 4.1 Update active metric runtime and TIGER runtime OpenSpec artifacts to describe adapter-based input resolution.
- [x] 4.2 Run focused tests for common metrics, TIGER metric runtime, metric callback attachment, and RKMeans config assertions.
- [x] 4.3 Run scoped ruff checks for touched source and tests.
- [x] 4.4 Run `openspec validate unify-metric-input-adapters --strict`.
