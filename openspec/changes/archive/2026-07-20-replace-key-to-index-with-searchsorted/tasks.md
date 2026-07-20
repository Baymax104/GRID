## 1. 重写 tensor_utils.py

- [x] 1.1 `load_model_output`：加载后用 `argsort` 排序 keys + predictions，用 `torch.unique` 检测重复 key 并 raise ValueError
- [x] 1.2 `gather_predictions_by_keys`：用 `torch.searchsorted` 查找行号，校验 `keys[indices] == lookup_keys`，不等则 raise KeyError；reshape 返回值保持原逻辑
- [x] 1.3 删除 `_build_key_to_index` 函数
- [x] 1.4 删除 `bundle["key_to_index"]` 运行时字段及懒构建逻辑

## 2. 验证

- [x] 2.1 `ruff check src/` 通过
- [x] 2.2 grep 确认零残留：`key_to_index`、`_build_key_to_index`
- [x] 2.3 确认函数签名不变（`load_model_output(file_path) -> dict`、`gather_predictions_by_keys(bundle, keys) -> Tensor`）
