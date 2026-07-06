## Context

当前 pipeline 存在两条相同的错误假设链路：

1. `sem_embeds_inference` 输出 embedding `.pt`，被 `rkmeans_train` / `rkmeans_inference` / `rqvae_train` / `rvq_train` 通过 `map_sparse_id_to_embedding()` 消费
2. `rkmeans_inference` 输出 semantic IDs `.pt`，被 `tiger_train` / `tiger_inference` 通过 `map_sparse_id_to_semantic_id()` 消费

这两条链路都把上游 `.pt` 视为“item_id 可直接索引的 tensor”。这使得推理端必须把 keyed predictions 强行投影成 ID-indexed tensor，而下游则通过 `tensor[item_id]` 或 `tensor.t()[item_id]` 直接取值。该设计隐藏了关键前提：item IDs 必须从 0 开始、连续、且最大 ID 接近样本数。实际数据不满足这个前提，因此在多进程 inference 收尾阶段触发越界。

用户已明确要求采用“keyed bundle + 显式 lookup”的正确协议，并接受旧协议产物全部失效。由此本次设计可以直接切断旧的 ID-indexed tensor 语义，而无需兼容双读。

## Goals / Non-Goals

**Goals:**
- 用单文件 keyed bundle 取代 ID-indexed tensor 协议
- 统一 `sem_embeds_inference` 与 `rkmeans_inference` 的输出格式
- 让所有下游消费者通过显式 key lookup 获取 embedding / semantic IDs
- 消除 `on_predict_end()` 中因错误协议导致的越界异常与退出假死

**Non-Goals:**
- 不兼容历史裸 tensor 产物
- 不保留“按 `max(item_id)+1` 开稠密 tensor”的过渡方案
- 不改变 experiment 参数名（`embedding_path` / `semantic_id_path` 保持不变）
- 不在本次内重构整个数据预处理框架，只在必要位置引入 keyed lookup 支撑

## Decisions

### D1: 继续沿用 `merged_predictions_tensor.pt` 文件名，但内容升级为单文件 bundle
- **选择**：文件名保持 `merged_predictions_tensor.pt`，内容改为：
  ```python
  {"keys": torch.Tensor, "predictions": torch.Tensor}
  ```
- **理由**：这样可以保持 shell 脚本、experiment 顶层参数、路径约定不变，把 breaking 面收敛到“内容协议”而不是“路径协议”。
- **备选**：改成两个文件（`keys.pt` + `predictions.pt`）——否决，配对关系容易丢失；改新文件名——否决，会扩大无意义的路径改动。

### D2: `keys` / `predictions` 统一字段名，不按具体任务命名
- **选择**：bundle 字段统一为 `keys` 与 `predictions`。
- **理由**：这样 `sem_embeds_inference` 与 `rkmeans_inference` 可以共用一套 loader / lookup 工具；避免 `item_ids` / `embeddings` / `semantic_ids` 这样的任务特化字段名把协议层和业务层耦合。

### D3: 上游只写新协议，不提供旧协议 fallback
- **选择**：所有新生成 `.pt` 都只写 keyed bundle；下游读取逻辑只接受 keyed bundle。
- **理由**：用户已接受旧产物全部失效；保留双协议只会延长过渡期并增加维护面。

### D4: lookup 基础设施放在 `tensor_utils.py`，业务接入点留在 `preprocessing.py`
- **选择**：
  - `src/utils/tensor_utils.py` 负责：bundle 组装、bundle 加载、key->row 索引构建、按 key lookup
  - `src/data/components/preprocessing.py` 继续作为 embedding / semantic id 映射的业务入口
- **理由**：把协议层工具与业务预处理分离，便于后续复用到更多 keyed prediction artifacts。

### D5: `sem_embeds_inference` 与 `rkmeans_inference` 必须同时升级
- **选择**：两条产物链一次性统一切换到新协议。
- **理由**：如果只修 `sem_embeds_inference`，则 `tiger_*` 仍绑定旧 semantic ID 协议；统一升级后整条 pipeline 语义一致。

## Risks / Trade-offs

- **[风险] 下游配置仍直接 `torch.load` 后把结果当 tensor 使用** → 缓解：引入显式 loader/adapter，修改相关 YAML 指向新 loader，而不是继续直接 `torch.load`
- **[风险] keyed lookup 若逐样本用 Python dict 循环，可能引入性能开销** → 缓解：在加载 bundle 时一次性构建 key->row index map，并缓存到 config/adapter 对象中，而非每条样本重复构建
- **[风险] semantic ID 当前形状依赖 `id_map[:num_hierarchies].t()[v]` 语义，改 lookup 容易出错** → 缓解：先明确 bundle 中 `predictions` 对 `rkmeans_inference` 的标准形状（推荐 `[N, H]`），再在 `map_sparse_id_to_semantic_id()` 中统一按 `predictions[row_indices]` 取值并 reshape
- **[权衡] 不兼容旧产物意味着所有旧实验缓存作废** → 可接受，用户已明确接受旧协议全部失效
