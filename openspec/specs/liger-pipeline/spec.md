# liger-pipeline Specification

## Purpose
将 LIGER 训练和推理接入统一 src.main 入口、组件配置及根脚本，明确公共 reader、writer 和上游产物的装配关系，并通过有界训练、恢复及预测验证确认可交付链路。
## Requirements
### Requirement: 统一入口与脚本
训练和推理 MUST 通过 src.main 与 Hydra experiment，脚本支持 dry-run、notes 两种形式、额外 override 后置及 wandb URI 引号。

#### Scenario: 统一入口与脚本验证
- **WHEN** 带空格 notes 和带等号 artifact URI
- **THEN** 实际 Bash 参数经 Hydra 解析正确，错误与空值被拒绝

### Requirement: 配置与产物
配置 MUST 复用公共 datamodule、metrics、writer、lineage；训练显式使用 training split 推导 seen 集合；文档区分 GRID 数据适配与原论文复现。

#### Scenario: 配置与产物验证
- **WHEN** 训练或推理 compose
- **THEN** 输入字段齐全、输出受 paths.output_dir 管理

### Requirement: 有界验证
交付 MUST 包含聚焦 CPU 测试、Hydra compose、Bash 验证、三个 strict OpenSpec 验证及本地可用 GPU dry run。完整训练 MUST 留给用户手动启动。

#### Scenario: 有界验证验证
- **WHEN** 执行 dry run
- **THEN** 统一入口完成一个训练更新和有界推理，无正式 W&B 结果发布
