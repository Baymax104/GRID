## Context

当前 `LocalPickleWriter` 继承 Lightning `BasePredictionWriter`，并配置 `write_interval: batch`。Lightning 在 prediction loop 中会为 `BasePredictionWriter` 准备 `batch_indices`，但当前项目的推理 dataloader 基于 `SequenceDataset(IterableDataset)` 和 `DataloaderWithIterationRetry`，Lightning 无法为该组合推断 batch indices，因此会发出 warning。

项目自己的 writer 并不使用 `batch_indices`。它的业务输出由 `predict_step` 返回的 `ModelOutput(keys, predictions)` 决定，最终在 `on_predict_end` 合并为 keyed prediction bundle。

## Goals / Non-Goals

**Goals:**

- 消除由 `BasePredictionWriter` batch index 推断触发的 warning。
- 保持现有 batch buffering、flush frequency、分布式 barrier、rank 0 merge、post-processing 行为。
- 将 writer API 收窄为当前实际支持的 batch-only 语义。

**Non-Goals:**

- 不修改 `DataloaderWithIterationRetry` 或 streaming dataset 架构。
- 不引入 map-style dataset 或 batch sampler。
- 不改变 `ModelOutput` 或 `merged_predictions_tensor.pt` bundle 协议。
- 不删除 validation/test metric 的 epoch-end 聚合逻辑。

## Decisions

1. `BaseBufferedWriter` 改为继承普通 `Callback`。

   这样 writer 不再被 Lightning prediction loop 识别为 `BasePredictionWriter`，Lightning 就不会为了该 callback 准备 `batch_indices`。替代方案是过滤 warning 或重构 dataloader；前者只是静音，后者修改面过大。

2. 使用 `on_predict_batch_end` 处理 batch outputs。

   `on_predict_batch_end` 能直接接收 `predict_step` 输出，满足现有 `handle_batch(ModelOutput)` 逻辑。最终 flush 和 merge 继续放在 `on_predict_end`，避免在 epoch end 保留整轮 predictions。

3. 删除 `write_interval` 和 `write_on_epoch_end` 语义。

   当前所有 inference 配置均使用 `write_interval: batch`，没有 epoch writer 使用者。删除该配置可以避免未来误配置为 `epoch` 或 `batch_and_epoch` 后得到隐藏的 `NotImplementedError`。

## Risks / Trade-offs

- [Risk] 如果外部未纳入仓库的配置仍传入 `write_interval`，实例化会失败。→ Mitigation: 更新仓库内全部 inference callback 配置，并通过测试覆盖构造函数签名。
- [Risk] writer 不再使用 Lightning 的 `BasePredictionWriter` 抽象。→ Mitigation: 当前业务并不需要 `batch_indices` 或 Lightning interval 语义，普通 `Callback` 更贴合实际职责。
- [Risk] epoch writer 行为被移除。→ Mitigation: 当前大规模 inference 依赖 batch flush 控制内存，epoch writer 不符合现有运行模型。
