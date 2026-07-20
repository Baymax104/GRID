# mainline-restart-decoupling Specification

## Purpose
TBD - created by archiving change deprecate-restart-mainline. Update Purpose after archive.
## Requirements
### Requirement: Default pipeline SHALL not depend on restart metadata
默认训练与推理主链路在初始化 pipeline 时，必须不再读取或依赖 restart metadata 文件，也不得因为 restart callback 或 launcher 的存在与否而改变默认行为。

#### Scenario: Train pipeline initialization without restart metadata
- **WHEN** 训练入口调用 `pipeline_launcher()` 初始化 pipeline
- **THEN** 主链路不得读取 `restart_metadata.json`
- **AND** 主链路不得导入或调用 restart metadata 读取工具来决定默认 `ckpt_path`

#### Scenario: Inference pipeline initialization without restart metadata
- **WHEN** 推理入口调用 `pipeline_launcher()` 初始化 pipeline
- **THEN** 主链路行为必须与 restart 机制无关
- **AND** 不得要求配置中存在 restart callback 或 metadata 路径

### Requirement: Checkpoint directory resolution SHALL remain available
当配置显式启用目录解析最新 checkpoint 的能力时，系统必须继续支持从 checkpoint 目录中自动选择最近的 `.ckpt` 文件。

#### Scenario: Resolve latest checkpoint from directory
- **WHEN** `ckpt_path` 指向目录且 `should_retrieve_latest_ckpt_path` 为 `true`
- **THEN** 系统必须选择最近的 checkpoint 文件作为运行时 `ckpt_path`
- **AND** 该行为不得依赖 restart metadata

### Requirement: Restart modules SHALL be marked deprecated
仓库保留的 restart 模块必须明确标注为废弃历史能力，说明其不属于默认主链路。

#### Scenario: Reader inspects restart module
- **WHEN** 开发者查看 `src/utils/restart_job.py` 或 `src/utils/restart_job_utils.py`
- **THEN** 文件或核心说明中必须明确标注 deprecated
- **AND** 注释必须说明默认 train / inference 主链路不再接入该机制

### Requirement: Default command templates SHALL not expose obsolete restart flags
默认训练命令模板不得继续包含已经失效的 restart 参数遗留。

#### Scenario: Inspect default training shell script
- **WHEN** 开发者查看仓库中的默认训练 shell 脚本
- **THEN** 脚本中不得包含 `+should_skip_retry=true`
- **AND** 脚本内容应只反映当前主链路真实需要的参数

