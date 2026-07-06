## Why

当前 `sem_embeds_inference` 与 `rkmeans_inference` 会在推理结束时把 `(item_id, prediction)` 列表后处理为 `merged_predictions_tensor.pt`。现有实现错误地假设 `item_id` 可直接作为稠密 tensor 的行下标，因此按 `len(data)` 分配输出张量并执行 `output_tensor[item_id] = prediction`。当 `item_id` 非连续或大于样本数时，推理结束阶段会抛出越界异常，导致 rank 0 在 `on_predict_end()` 的重试逻辑中反复重试，表现为“预测完成后程序无法退出”。

更根本的问题是：把“业务主键”伪装成“稠密数组下标”本身就是错误协议。正确的产物应该显式保留 `keys` 与 `predictions` 的对齐关系，由下游在消费时建立 `item_id -> row offset` 映射，而不是继续依赖隐式的 ID-indexed tensor 语义。

用户已明确接受 **不兼容旧协议**，旧的裸 tensor 产物全部失效。因此本次变更可以直接升级整条 pipeline 的中间产物协议，而无需保留兼容读取分支。

## What Changes

- **BREAKING（内部 pipeline 协议）** 将 `merged_predictions_tensor.pt` 的内容从“裸 `torch.Tensor`，按 item_id 直接索引”改为单文件 keyed prediction bundle：
  ```python
  {
      "keys": torch.Tensor,
      "predictions": torch.Tensor,
  }
  ```
- `sem_embeds_inference` 与 `rkmeans_inference` 均输出新协议；文件名继续沿用 `merged_predictions_tensor.pt`
- 删除“按 key 直接 scatter 到 tensor 行号”的 merge 逻辑，改为按预测顺序紧凑堆叠 `keys` 与 `predictions`
- 下游消费方不再把 `.pt` 当作可直接按 item_id 索引的 tensor，而是通过显式 lookup 获取对应 embedding / semantic IDs
- 明确不兼容旧协议：不提供对历史裸 tensor 产物的 fallback 读取

## Capabilities

### New Capabilities
- `keyed-prediction-bundle-artifact`: 规定推理阶段的 keyed 预测产物必须以 `{keys, predictions}` 单文件 bundle 形式保存

### Modified Capabilities
- `data-precomputed-lookup`: 下游数据预处理从“直接 tensor 索引”改为“显式 key lookup”消费预计算 embedding / semantic IDs

## Impact

- 受影响上游：`src/utils/inference_utils.py`、`src/utils/tensor_utils.py`，以及使用 `LocalPickleWriter` 生成 `.pt` 的推理实验
- 受影响下游代码：`src/data/components/preprocessing.py` 中的 `map_sparse_id_to_embedding()` 与 `map_sparse_id_to_semantic_id()`
- 受影响配置：`configs/data/rkmeans_train.yaml`、`rkmeans_inference.yaml`、`rqvae_train.yaml`、`rvq_train.yaml`、`tiger_train.yaml`、`tiger_inference.yaml`
- 受影响文档/约定：`README.md`、`AGENTS.md`、相关 shell 脚本中对 `merged_predictions_tensor.pt` 的语义描述
- 旧产物全部失效，需要重新跑 `sem_embeds_inference` / `rkmeans_inference` 生成新协议文件
