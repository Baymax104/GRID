## ADDED Requirements

### Requirement: Main config entrypoints SHALL remain thin
`train.yaml` 与 `inference.yaml` 必须作为 defaults 导入层与通用运行开关层，而不再承担 experiment 级手动输入入口职责。

#### Scenario: User chooses an experiment config
- **WHEN** 用户通过 `experiment=...` 启动训练或推理
- **THEN** 主要手动输入字段必须由 experiment 配置提供
- **AND** `train.yaml` / `inference.yaml` 不得再次暴露同类实验级字段

### Requirement: Experiment configs SHALL own manual data/checkpoint inputs
`data_dir`、`ckpt_path` 等实验级手动输入必须统一由 experiment 顶层提供。

#### Scenario: Inference experiment provides checkpoint path
- **WHEN** 某个 inference experiment 需要 checkpoint 路径
- **THEN** 该路径必须在 experiment 顶层定义

#### Scenario: Data directory is provided for any experiment
- **WHEN** 用户配置 experiment 的数据目录
- **THEN** `data_dir` 必须在 experiment 顶层定义
- **AND** 路径默认层只能透传该值

### Requirement: Thin main entrypoints SHALL preserve official experiment behavior
在主入口瘦身之后，官方 experiment 的配置合成与主链路行为必须保持可用。

#### Scenario: Official experiment composes after main entrypoint thinning
- **WHEN** 官方 experiment 与 `train.yaml` 或 `inference.yaml` 组合
- **THEN** 配置必须仍可被 Hydra 正常合成
