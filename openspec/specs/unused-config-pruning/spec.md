# unused-config-pruning Specification

## Purpose
TBD - created by archiving change remove-unused-config-stubs. Update Purpose after archive.
## Requirements
### Requirement: Unreferenced config stubs SHALL be removable
对于已经确认未被主配置链路、experiment、脚本或文档引用的配置模板，系统必须允许将其从仓库中删除，而不影响默认配置装配。

#### Scenario: Remove unused callback config stub
- **WHEN** 一个 callback 配置文件未被 defaults、experiment、脚本或文档引用
- **THEN** 该文件可以被删除
- **AND** 默认 train / inference 配置装配不得因此失败

#### Scenario: Remove unused trainer config stub
- **WHEN** 一个 trainer 子配置文件未被任何配置链路引用
- **THEN** 该文件可以被删除

### Requirement: Empty local config placeholder MAY be removed
如果 `configs/local/` 仅包含未被使用的占位文件，则本轮清理可以移除该目录占位。

#### Scenario: Remove empty local config directory placeholder
- **WHEN** `configs/local/` 只包含 `.gitkeep` 且无任何引用
- **THEN** 该占位文件可以被删除

### Requirement: Pruning SHALL not affect active config files
删除死配置时，不得影响仍在主链路中使用的配置模板与 experiment 配置。

#### Scenario: Active config templates remain untouched
- **WHEN** 执行死配置清理
- **THEN** `train.yaml`、`inference.yaml`、`callbacks/default.yaml`、`logger/default.yaml`、`trainer/default.yaml` 等主配置文件必须保持可用

