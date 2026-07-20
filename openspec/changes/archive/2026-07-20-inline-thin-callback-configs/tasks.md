## 1. 内联 inference 薄包装配置

- [x] 1.1 将 `configs/callbacks/inference_default.yaml` 的默认 callback 逻辑直接并入更直接的 inference 配置入口
- [x] 1.2 删除 `configs/callbacks/inference_default.yaml`

## 2. 收敛 progress bar 薄包装配置

- [x] 2.1 评估并实施 `configs/callbacks/one_based_tqdm_progress_bar.yaml` 的最小内联方案
- [x] 2.2 若已完成内联，则删除 `configs/callbacks/one_based_tqdm_progress_bar.yaml`

## 3. 验证

- [x] 3.1 验证 `train.yaml` 与 `inference.yaml` 默认 callback 行为保持不变
- [x] 3.2 全文搜索确认已删除薄包装配置没有悬挂引用
