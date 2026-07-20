## 1. 删除已确认未引用的配置文件

- [x] 1.1 删除 `configs/callbacks/local_pickle_writer.yaml`
- [x] 1.2 删除 `configs/callbacks/rich_progress_bar.yaml`
- [x] 1.3 删除 `configs/trainer/ddp.yaml`

## 2. 删除空本地配置占位

- [x] 2.1 删除 `configs/local/.gitkeep`
- [x] 2.2 确认 `configs/local/` 不再作为有效配置目录参与本轮维护

## 3. 验证

- [x] 3.1 全文搜索确认已删除文件没有悬挂引用
- [x] 3.2 复核主配置入口（train/inference/default callbacks/logger/trainer）保持不变
