## Why

用户要求将商品表示扩为256维并完全去掉adapter，验证独立内容模型的原始能力。此前128维adapter在2k/10k未过增量门槛，10k相对2k两组NLL改善。

## What Changes

- 新增固定PCA256、隐藏/查询256维、无adapter的单次10k训练。
- 配对既有128固定组，复用同缓存、预算和512开发用户。
- 保留旧128训练与checkpoint兼容，不启动真实实验。

## Capabilities

### New Capabilities
- `fixed-content-capacity`: 无适配器的容量扩展资格。

## Impact

内容网络、训练契约、容量评价、配置与现有脚本的experiment override。
