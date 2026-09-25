# active-research-method-surface Specification

## Purpose
TBD - created by archiving change prune-retired-research-methods. Update Purpose after archive.
## Requirements
### Requirement: Repository SHALL expose only active research methods
仓库的可执行 recommendation 方法 SHALL 限于 TIGER、LIGER 与 CoPMRec；已结题的 BRIR、MIR/item-resolution、CGBS/catalog-grounded 和 training probe SHALL NOT 保留可导入实现或 Hydra experiment 入口。

#### Scenario: Enumerate recommendation packages
- **WHEN** 检查 `src/recommendation/` 下的可导入方法目录
- **THEN** 只存在 TIGER、LIGER 及其共享基础实现

#### Scenario: Compose experiment entrypoints
- **WHEN** 枚举 `configs/experiment/` 中的 recommendation 训练与推理入口
- **THEN** 不得出现已结题方法的 experiment 配置

### Requirement: CoPMRec SHALL retain its frozen mechanism surface
CoPMRec SHALL 保留 mass 主方法以及 legal generation、max aggregation 和 content-only 控制，并 SHALL NOT 暴露深度条件聚合、动态门控、学习排序、候选并集、偏好分散或来源保护入口。

#### Scenario: Instantiate CoPMRec main method
- **WHEN** Hydra 装配 `liger_joint_train` 或 `liger_joint_inference`
- **THEN** 模型使用合法 SID 树上的 learned mass probability mixture

#### Scenario: Reject retired mechanism controls
- **WHEN** 调用方请求已淘汰的深度条件或动态机制
- **THEN** 配置装配或模型初始化必须失败，且仓库中不存在对应专用启动脚本

### Requirement: Active checkpoints SHALL remain load-compatible
清理 SHALL 保持当前 LIGER 与 CoPMRec checkpoint 的输入身份检查和参数路径兼容。

#### Scenario: Load current CoPMRec checkpoint state
- **WHEN** checkpoint 包含 `dynamic_gate.bias` 与 `liger_joint_mixture` 标记
- **THEN** `JointMixtureLiger` 能严格加载该 state dict 并执行固定 mass mixture

### Requirement: Retired method names SHALL be absent from executable surfaces
结构回归检查 SHALL 扫描源码 import、Hydra `_target_`、根目录脚本和测试，确保已淘汰方法名称不再形成可执行入口。

#### Scenario: Run retired-surface regression
- **WHEN** 执行仓库结构测试
- **THEN** BRIR、MIR/item-resolution、CGBS/catalog-grounded、training probe 及已结题 LIGER 变体均无残留可执行引用
