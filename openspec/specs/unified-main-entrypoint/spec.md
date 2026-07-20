# unified-main-entrypoint Specification

## Purpose
TBD - created by archiving change unify-main-entrypoint. Update Purpose after archive.
## Requirements
### Requirement: Official experiments SHALL declare run mode explicitly
每个 official experiment 必须显式声明自己的运行模式，而不能再依赖 Python 入口名称或主配置文件名称隐式决定。

#### Scenario: Training experiment declares mode
- **WHEN** 官方训练 experiment 被定义
- **THEN** 该 experiment 必须在顶层声明 `run_mode: train`

#### Scenario: Inference experiment declares mode
- **WHEN** 官方推理 experiment 被定义
- **THEN** 该 experiment 必须在顶层声明 `run_mode: inference`

### Requirement: Repository SHALL provide a single Python main entrypoint
仓库必须提供单一 Python 运行入口来承载 train / inference 主链路分发。

#### Scenario: User launches training through unified entrypoint
- **WHEN** 用户通过统一入口执行一个 `run_mode: train` 的 experiment
- **THEN** 系统必须执行训练链路
- **AND** 可按配置选择是否继续执行测试链路

#### Scenario: User launches inference through unified entrypoint
- **WHEN** 用户通过统一入口执行一个 `run_mode: inference` 的 experiment
- **THEN** 系统必须执行推理链路
- **AND** 不得进入训练链路

### Requirement: Main configuration SHALL be unified and thin
主配置文件必须统一为单一入口层，并只承担 defaults 导入与通用运行开关职责。

#### Scenario: Unified main config composes an experiment
- **WHEN** 官方 experiment 与单一主配置文件组合
- **THEN** 主配置文件不得再次承担 train/inference 分流语义
- **AND** train/inference 差异必须由 experiment 的 `run_mode` 和本地配置决定

### Requirement: Legacy entrypoints SHALL be removed
旧的 `src/train.py` 与 `src/inference.py` 必须被移除，以避免继续暴露双入口语义。

#### Scenario: Repository source layout after migration
- **WHEN** 开发者查看运行入口源码
- **THEN** 仓库中必须只保留统一主入口文件

### Requirement: Command templates SHALL use the unified entrypoint
仓库中的默认脚本与文档命令模板必须切换到统一入口。

#### Scenario: Inspect shell scripts and docs
- **WHEN** 开发者查看仓库中的 `*.sh`、`README.md` 或 `AGENTS.md`
- **THEN** 默认命令模板必须使用统一入口

