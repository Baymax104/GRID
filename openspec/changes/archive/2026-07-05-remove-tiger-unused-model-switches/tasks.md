## 1. 删除无效模型开关

- [x] 1.1 更新 `src/common/modules/transformer_base_module.py`，移除 `compile` 与 `weight_tying` 构造参数
- [x] 1.2 删除 `TransformerBaseModule` 中与 `compile` 相关的死代码/注释残留
- [x] 1.3 将基类 `get_embedding_table()` 固定为返回 encoder input embeddings

## 2. 清理官方 TIGER 配置

- [x] 2.1 更新 `configs/experiment/tiger_train.yaml`，删除 `weight_tying` 与 `compile`
- [x] 2.2 更新 `configs/experiment/tiger_inference.yaml`，删除 `weight_tying` 与 `compile`

## 3. 验证

- [x] 3.1 全文检查确认仓库中不再保留官方 `weight_tying` / `compile` 配置入口
- [x] 3.2 最小验证相关 Python 与 YAML 文件仍可解析
