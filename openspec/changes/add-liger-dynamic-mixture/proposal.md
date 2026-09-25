## Why

用户要求先尝试动态混合概率，sampling 仅保留想法。固定混合相对原始 LIGER 有正向证据；本次只检验前缀状态是否能预测两路概率的相对有效性，不恢复学习排序和 dense20 联合池。

## What Changes

- 增加零初始化的前缀级 sigmoid 门控与全局可学习常数对照，冻结 LIGER。
- training split 最后目标的 teacher-forcing 紧凑缓存仅保存到执行主机本地；内部用户哈希留出选择 checkpoint。
- 推理保持 beam20、cold 补充及原内容重排，记录来源和门控身份。
- 准备统一入口、聚焦测试和有边界手动实验命令，不自动运行实验。

## Capabilities

### New Capabilities
- `liger-dynamic-mixture`: 内容与生成条件概率的前缀级门控、训练缓存及来源验证。

### Modified Capabilities

无；固定混合行为保持兼容。

## Impact

涉及 LIGER 概率处理器、轻量门控模块、共享 writer、data helper、配置和根目录脚本；无新增依赖。
