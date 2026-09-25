## Why

建立以 LIGER 为主要 baseline 的正式比较，需要可核查的官方方法适配，不能沿用 CGBS hybrid 冒充。

## What Changes

依赖前两个提案；新增薄 experiment/component 配置、训练与推理 Bash 入口、文档及 CPU 测试；统一入口执行本地 GPU dry run。

## Capabilities

### New Capabilities

- `liger-pipeline`：LIGER 配置、入口与验证。

### Modified Capabilities

无。

## Impact

configs/、liger_train.sh、liger_inference.sh、scripts/liger_common.sh、src/data/components/liger.py、../../../../research/docs/grid-experiments/、tests/。真实 dry run 同时修复公共 TFRecordReader 的 Windows file URI 盘符识别。保留现有用户改动，不启动完整实验。
