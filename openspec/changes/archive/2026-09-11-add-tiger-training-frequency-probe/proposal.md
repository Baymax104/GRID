## Why

固定候选配额在 Beauty/RKMeans 开发设置未通过效用门槛，但 teacher forcing 显示 Tail 第 2–3 层条件评分不足。需要隔离 baseline 的小规模训练侧探针，验证条件分支重加权能否优于同预算普通 CE 微调，而不是直接扩展完整实验。

## What Changes

- 新增独立训练模块，仅改变训练目标，复用原 Tiger 数据流、forward、评估与生成。
- 按现有子序列随机展开规则计算训练目标的期望次数，建立带来源指纹的条件分支统计。
- 新增受限、归一化的第 2–3 层频次权重；普通 CE 对照与干预共用相同初始化和训练预算。
- 显式区分权重初始化与训练恢复，保留 checkpoint 参数兼容性。
- 新增薄配置、根启动脚本、单元测试及手动执行方案；不自动启动完整训练。

## Capabilities

### New Capabilities

- `tiger-training-frequency-probe`: 独立训练探针、期望训练频次、权重初始化、可审计的人工验证契约。

### Modified Capabilities

无。原 baseline 训练、数据、解码与验证契约保持不变。

## Impact

新增 `src/recommendation/tiger_training_probe/`、data 统计 helper、独立配置与脚本；复用共享 Artifact resolver、writer 与 lineage callback。不增加依赖，不修改 Tiger、dataset、collate、decoder 或统一入口。
