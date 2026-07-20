## 1. 升级 keyed prediction artifact 协议

- [x] 1.1 在 `src/utils/tensor_utils.py` 中新增 keyed bundle 组装/加载/lookup 工具，约定 bundle 结构为 `{keys, predictions}`
- [x] 1.2 删除/替换旧的 `merge_list_of_keyed_tensors_to_single_tensor()` 的 ID-indexed tensor 语义，不再按 `item_id` scatter 到行号
- [x] 1.3 `src/utils/inference_utils.py`：`LocalPickleWriter._merge_files()` 改为保存单文件 keyed bundle 到 `merged_predictions_tensor.pt`

## 2. 同步升级上游推理实验输出

- [x] 2.1 确认 `sem_embeds_inference` 的 `.pt` 输出为 keyed bundle，`keys` 与 embedding predictions 长度一致
- [x] 2.2 确认 `rkmeans_inference` 的 `.pt` 输出为 keyed bundle，`keys` 与 semantic ID predictions 长度一致
- [x] 2.3 删除对旧裸 tensor 协议的任何写入或读取 fallback

## 3. 改造下游 embedding 消费链

- [x] 3.1 更新 `src/data/components/preprocessing.py::map_sparse_id_to_embedding()`，改为通过 keyed lookup 获取 embedding
- [x] 3.2 更新 `configs/data/rkmeans_train.yaml`、`rkmeans_inference.yaml`、`rqvae_train.yaml`、`rvq_train.yaml`，使其加载 keyed bundle 所需对象，而非把 `torch.load(...)` 结果直接当 tensor

## 4. 改造下游 semantic ID 消费链

- [x] 4.1 更新 `src/data/components/preprocessing.py::map_sparse_id_to_semantic_id()`，改为通过 keyed lookup 获取 semantic IDs
- [x] 4.2 更新 `configs/data/tiger_train.yaml`、`tiger_inference.yaml`，使其加载 keyed bundle 所需对象，而非继续依赖 ID-indexed tensor

## 5. 文档与约定同步

- [x] 5.1 更新 `README.md`、`AGENTS.md` 中对 `merged_predictions_tensor.pt` 的描述，明确其为 keyed bundle 而非裸 tensor
- [x] 5.2 复查 `*.sh` 与相关说明文档，确保路径不变但语义描述与新协议一致

## 6. 验证收尾

- [x] 6.1 最小验证 `sem_embeds_inference` 输出 bundle 的 `keys/predictions` 对齐关系
- [x] 6.2 最小验证 `rkmeans_train` / `rkmeans_inference` 可从 bundle 正确 lookup embedding
- [x] 6.3 最小验证 `tiger_train` / `tiger_inference` 可从 bundle 正确 lookup semantic IDs
- [x] 6.4 复查仓库（排除 `openspec/`）确认不存在继续把 `merged_predictions_tensor.pt` 视为裸 tensor 的调用点
