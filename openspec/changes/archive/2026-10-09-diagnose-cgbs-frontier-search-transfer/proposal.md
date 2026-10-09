## Why
global/branch验证frontier CE下降但推荐指标退化，需要区分实现不一致与离线竞争目标无法转移到在线搜索。用户已授权一次有界诊断，不新增训练或调参。
## What Changes
- 固定evaluation的256用户，一次analysis作业比较源A、global和branch的500步头。
- 检查off/零残差复现、固定A候选目标rank/margin、在线路径存活、候选偏移和残差尺度。
- 使用共享checkpoint入口、Trainer.test和共享writer；有效性失败仅发布失败诊断，不产生机制通过结论。
## Capabilities
### New Capabilities
- `cgbs-frontier-search-transfer`: 固定checkpoint的有界目标/搜索转移诊断。
### Modified Capabilities
无。
## Impact
新增诊断模型、test datamodule、汇总writer、配置与根脚本；不改变训练模型或搜索算法，不增加依赖。
