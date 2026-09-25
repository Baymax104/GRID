## ADDED Requirements

### Requirement: Retired research configs SHALL be removed as complete slices
当研究状态将方法标记为结题且当前实验矩阵不再引用时，系统 SHALL 同时删除该方法的 experiment、model、data、callback、trainer 配置和启动脚本，而 SHALL NOT 留下可被 Hydra 单独装配的残余配置。

#### Scenario: Prune a retired method
- **WHEN** 方法不属于当前 CoPMRec/LIGER/TIGER 实验矩阵
- **THEN** 该方法所有专用配置组与脚本均不存在
- **AND** 活跃 experiment 的 defaults 仍能完成装配
