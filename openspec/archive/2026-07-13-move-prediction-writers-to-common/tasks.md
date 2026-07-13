## 1. 移动文件

- [x] 1.1 将 `src/utils/inference_utils.py` 移至 `src/common/components/prediction_writers.py`（mv），文件内容不变

## 2. 更新 Hydra _target_ 字符串引用

- [x] 2.1 `src/utils/launcher_utils.py`：将 `DRY_RUN_DISABLED_CALLBACK_TARGETS` 中的 `"src.utils.inference_utils.LocalPickleWriter"` 改为 `"src.common.components.prediction_writers.LocalPickleWriter"`
- [x] 2.2 `configs/callbacks/sem_embeds_inference.yaml`：`_target_: src.utils.inference_utils.LocalPickleWriter` → `_target_: src.common.components.prediction_writers.LocalPickleWriter`
- [x] 2.3 `configs/callbacks/rkmeans_inference.yaml`：同上
- [x] 2.4 `configs/callbacks/tiger_inference.yaml`：同上

## 3. 验证

- [x] 3.1 `uv run ruff check src/` — 确认无未解析导入和 lint 错误（仅剩 1 个预先存在的 B008）
- [x] 3.2 grep 确认 `src.utils.inference_utils` 零残留（代码和配置中无残留，仅 openspec 文档中有引用）
- [x] 3.3 grep 确认 `inference_utils` 文件不再存在于 `src/utils/` 中
