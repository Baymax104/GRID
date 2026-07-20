## 1. CLI 入口与配置开关

- [x] 1.1 在 `src/train.py` 中实现 `--dry-run` 预处理，并转换为内部 `dry_run=true`
- [x] 1.2 在 `src/inference.py` 中实现 `--dry-run` 预处理，并转换为内部 `dry_run=true`
- [x] 1.3 为默认配置提供 `dry_run` 开关的安全默认值，确保非 dry run 行为不变

## 2. Dry run 主链路行为

- [x] 2.1 在装配阶段为训练 dry run 注入最小 trainer 覆盖（单 step / 单 batch，禁用 val/test）
- [x] 2.2 在装配阶段为推理 dry run 注入最小 predict batch 覆盖
- [x] 2.3 在 dry run 下禁用写入型 logger（CSV / W&B）与 checkpoint callback
- [x] 2.4 在 dry run 下禁用推理结果写入 callback，确保不生成 pickle / tensor 产物

## 3. 验证

- [x] 3.1 验证 `--dry-run` 与 Hydra 参数可共存，不会触发 CLI 解析错误
- [x] 3.2 验证训练 dry run 不写 checkpoint、CSV、W&B 结果
- [x] 3.3 验证推理 dry run 不写 prediction pickle / `merged_predictions_tensor.pt`
- [x] 3.4 验证 dry run 仍保留 Hydra 输出目录与普通运行日志
