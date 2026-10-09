## Why

BMX-31/32/33 需要逐层目标路径证据；现有 candidate trace 仅保留最终候选，无法判定首次丢失层。新增只观察实际解码的路径记录，使三种控制能做配对路径分析。

## What Changes

- 默认关闭的 path_trace，记录实际 beam 前缀、目标存活、首次丢失层和可达父节点的分支概率。
- 单独的 keyed 路径 bundle，由共享辅助 writer 发布 node1 file reference。
- 更新三臂命令；旧 mass 仅复用最终结果，路径证据需要手动补采集。

## Capabilities

### New Capabilities
- `liger-path-trace`: 不改变解码结果的逐层路径采集与校验。

### Modified Capabilities
无。

## Impact

LIGER 生成入口、新增观察器/校验器、配置、聚焦测试及 Linear 命令；无新依赖、无训练与 checkpoint 参数变更。
