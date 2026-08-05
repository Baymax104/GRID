## Context

`Evaluator` 和 `SIDRetrievalEvaluator` 曾同时负责指标集合管理、输入转换和 reset/device 迁移。新框架已将这些职责拆分到 `MetricCallback`、`MetricEngine` 和 `SIDRetrievalMetricGroup`。

## Goals / Non-Goals

**Goals:**

- 删除旧 evaluator 包装层。
- 保留基础 retrieval metric 类供新 group 使用。
- 确保配置和模型不再引用旧 evaluator。

**Non-Goals:**

- 不改变 NDCG/Recall 计算公式。
- 不移动 `eval_metrics.py` 中的基础 metric 类。

## Decisions

### 删除 wrapper，保留 metric

只删除 `Evaluator` / `SIDRetrievalEvaluator`，因为它们代表旧生命周期框架；`NDCG` / `Recall` 是实际 metric implementation，仍由新 group 复用。

## Risks / Trade-offs

- [Risk] 外部未纳入仓库的脚本可能引用旧 evaluator。  
  Mitigation: 官方 configs 和源码均迁移到新 runtime；如需兼容外部脚本，可由调用方改用 `SIDRetrievalMetricGroup`。
