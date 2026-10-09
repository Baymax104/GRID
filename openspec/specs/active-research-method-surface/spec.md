# active-research-method-surface Specification

## Purpose
约束当前推荐方法、正式 CoPMRec 运行面及 checkpoint 身份边界，避免历史实现归档后重新成为活动要求。
## Requirements
### Requirement: Repository SHALL expose only active research methods
仓库的可执行 recommendation 方法 SHALL 限于 TIGER、LIGER、CoPMRec、LETTER 与 SASRec；已结题的 BRIR、MIR/item-resolution、CGBS/catalog-grounded 和 training probe SHALL NOT 保留可导入实现或 Hydra experiment 入口。

#### Scenario: Enumerate recommendation packages
- **WHEN** 检查 `src/recommendation/` 下的可导入方法目录
- **THEN** 只存在上述五种方法及其共享基础实现

#### Scenario: Compose experiment entrypoints
- **WHEN** 枚举 `configs/experiment/` 中的 recommendation 训练与推理入口
- **THEN** 不得出现已结题方法的 experiment 配置

### Requirement: CoPMRec SHALL retain its frozen mechanism surface
CoPMRec SHALL 使用正式 v2 联合视图与 learned mass probability mixture，等权优化 SID CE、joint catalog CE 和 mixture NLL，所有阶段不排除历史商品；消融入口 SHALL 仅支持 no_mixture、no_residual 和 no_joint_ce，不暴露旧 native view、fixed-alpha、content-only 或 legal/max 机制控制。

#### Scenario: Instantiate CoPMRec main method
- **WHEN** Hydra 装配 `copmrec_train` 或 `copmrec_inference`
- **THEN** 使用 `src.recommendation.copmrec.CoPMRec` 和正式 v2 契约，保留合法 SID 树上的 learned mass probability mixture

#### Scenario: Reject retired mechanism controls
- **WHEN** 调用方请求旧 `liger_joint_*` 入口或已淘汰的机制控制
- **THEN** 配置装配或模型初始化必须失败，且仓库中不存在对应专用启动脚本

### Requirement: Active checkpoints SHALL remain load-compatible
清理 SHALL 保持当前 LIGER 与正式 CoPMRec v2 checkpoint 的输入身份检查和严格恢复；旧 CoPMRec 开发 checkpoint SHALL NOT 被改标签当作当前正式版本恢复。

#### Scenario: Load current CoPMRec checkpoint state
- **WHEN** checkpoint 来自相同正式版本、损失契约与输入目录
- **THEN** 当前模型严格恢复状态并保持预测一致；旧版本或身份不匹配的 checkpoint 被拒绝

### Requirement: Retired method names SHALL be absent from executable surfaces
结构回归检查 SHALL 扫描源码 import、Hydra `_target_`、根目录脚本和测试，确保已淘汰方法名称不再形成可执行入口。

#### Scenario: Run retired-surface regression
- **WHEN** 执行仓库结构测试
- **THEN** BRIR、MIR/item-resolution、CGBS/catalog-grounded、training probe 及已结题 LIGER 变体均无残留可执行引用
