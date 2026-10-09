## Why

f91njtjx 的零更新诊断发现排序梯度可能压过基础检索目标，用户授权验证等权是否为主要问题。当前 ranking 权重硬编码为 1，无法进行单变量匹配对照。

## What Changes

- 将 ranking loss 权重配置化，默认 1，保留 v1/v1.1 旧行为与 checkpoint。
- 恢复时权重差异默认拒绝；仅显式启用重新加权时允许，记录来源权重与步数，其他契约仍严格检查。
- 用 training-only 梯度校准和两组固定 1000 更新的短程续训验证优化强度与推荐方向，不新增完整训练预算。

## Capabilities

### New Capabilities

- `copmrec-loss-weight`: 排序目标权重、恢复契约与有界验证。

### Modified Capabilities

无。

## Impact

修改 relevance 模型、对应组件配置及聚焦测试。续训走现有统一入口，不增加训练 runner，不冻结推荐模块，不停止用户正在运行的训练。
