# unused-config-pruning Specification

## Purpose
TBD - created by archiving change remove-unused-config-stubs. Update Purpose after archive.
## Requirements
### Requirement: Unreferenced config stubs SHALL be removable
对于已经确认未被主配置链路、experiment、脚本或文档引用的配置模板，系统 SHALL 允许将其从仓库中删除，而不影响默认配置装配。

#### Scenario: Remove unused callback config stub
- **WHEN** 一个 callback 配置文件未被 defaults、experiment、脚本或文档引用
- **THEN** 该文件可以被删除
- **AND** 默认 train / inference 配置装配不得因此失败

#### Scenario: Remove unused trainer config stub
- **WHEN** 一个 trainer 子配置文件未被任何配置链路引用
- **THEN** 该文件可以被删除

### Requirement: Empty local config placeholder SHALL be removable
如果 `configs/local/` 仅包含未被使用的占位文件，则本轮清理 SHALL 允许移除该目录占位。

#### Scenario: Remove empty local config directory placeholder
- **WHEN** `configs/local/` 只包含 `.gitkeep` 且无任何引用
- **THEN** 该占位文件可以被删除

### Requirement: Pruning SHALL not affect active config files
删除死配置时，SHALL NOT 影响仍在主链路中使用的配置模板与 experiment 配置。

#### Scenario: Active config templates remain untouched
- **WHEN** 执行死配置清理
- **THEN** `train.yaml`、`inference.yaml`、`callbacks/default.yaml`、`logger/default.yaml`、`trainer/default.yaml` 等主配置文件必须保持可用

### Requirement: Retired research configs SHALL be removed as complete slices
当研究状态将方法标记为结题且当前实验矩阵不再引用时，系统 SHALL 同时删除该方法的 experiment、model、data、callback、trainer 配置和启动脚本，而 SHALL NOT 留下可被 Hydra 单独装配的残余配置。

#### Scenario: Prune a retired method
- **WHEN** 方法不属于当前 CoPMRec/LIGER/TIGER 实验矩阵
- **THEN** 该方法所有专用配置组与脚本均不存在
- **AND** 活跃 experiment 的 defaults 仍能完成装配

