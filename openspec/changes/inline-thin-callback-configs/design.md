## Context

在清理掉明显未引用的死配置之后，`configs/` 中仍有一类“薄包装配置”存在：它们并非无效，但只承担非常轻的转发作用。例如 `inference_default.yaml` 只是为 inference 包一层默认 callback 组合，而 `one_based_tqdm_progress_bar.yaml` 只承载单个 callback 目标。这类文件会增加配置跳转层级，但当前仓库里没有带来明显复用收益。

本轮目标是做最小的结构收缩：以内联方式减少 callback 相关薄包装文件数量，同时保证 train / inference 当前行为保持一致。

## Goals / Non-Goals

**Goals:**
- 内联 `inference_default.yaml` 这类仅用于转发 defaults 的薄包装配置。
- 评估并收敛 `one_based_tqdm_progress_bar.yaml` 的薄包装层。
- 保持 train / inference 现有 callback 行为不变。

**Non-Goals:**
- 不重构 experiment 内联 callback 定义。
- 不删除 `callbacks/default.yaml`、`model_checkpoint.yaml`、`early_stopping.yaml`、`model_summary.yaml` 等仍有模板职责的文件。
- 不收缩 logger/trainer/paths 的重复配置。

## Decisions

### 1. 优先内联 inference 侧的薄包装入口
- 决策：优先处理 `configs/callbacks/inference_default.yaml`，将其承担的默认 progress bar 选择直接并入 `configs/inference.yaml` 或更直接的 callback 入口。
- 原因：它只被 `inference.yaml` 单点引用，且仅转发一层 defaults。

### 2. 谨慎处理单 callback 薄包装文件
- 决策：对 `one_based_tqdm_progress_bar.yaml` 采用“若能不破坏 Hydra 组织语义则内联，否则保留”的策略。
- 原因：它虽然很薄，但仍然是 callback 目标的命名锚点；是否完全移除取决于最小改动路径。
- 备选方案：保留它不动，仅删除 `inference_default.yaml`。若实现中发现完全内联会让配置更绕，应退回此方案。

## Risks / Trade-offs

- [过度内联使 callback 组织更难读] → 优先减少一层无意义 defaults 包装，不追求把所有 callback 文件压成单文件。
- [Hydra defaults 结构变化引入行为偏差] → 做 YAML 解析与默认入口复核，确保 train / inference 行为不变。

## Migration Plan

1. 先内联 `inference_default.yaml`。
2. 视实际实现复杂度决定是否同时去掉 `one_based_tqdm_progress_bar.yaml`。
3. 验证 `train.yaml` / `inference.yaml` 与 callback 默认装配保持一致。

## Open Questions

- `one_based_tqdm_progress_bar.yaml` 是否一起移除，取决于实现时是否能在不增加配置混乱的前提下完成内联。
