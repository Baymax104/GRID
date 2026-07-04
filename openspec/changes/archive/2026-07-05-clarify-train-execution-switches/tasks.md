## 1. 收敛训练 experiment 运行开关

- [x] 1.1 更新 `src/main.py` 训练链路，删除对 `cfg.train` 的依赖，并改为读取 `run_test_after_training`
- [x] 1.2 更新 dry-run 相关训练链路覆写逻辑，改为关闭 `run_test_after_training`

## 2. 迁移官方训练 experiment 配置

- [x] 2.1 更新所有 `configs/experiment/*_train.yaml`，删除 `train` 字段
- [x] 2.2 将训练 experiment 中的 `test` 字段统一重命名为 `run_test_after_training`
- [x] 2.3 复核训练 experiment 注释与 section 说明，使新字段语义清晰可读

## 3. 验证

- [x] 3.1 全文检查确认官方训练 experiment 不再包含顶层 `train:` 或旧的训练语义 `test:` 开关
- [x] 3.2 最小验证训练/推理链路代码对 `run_mode` 与 `run_test_after_training` 的消费关系保持一致
