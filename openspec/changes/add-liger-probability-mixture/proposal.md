## Why
用户确认在LIGER上发展主方法并继续实施。max势差干预已确认相对合法性控制有增量，但相对dense退化，且仅保留原278独有命中的68个；需有边界地验证条件概率混合能否改善这个取舍。

## What Changes
- 增加合法条件归一化控制与精确内容分支概率混合，默认不启用。
- 固定alpha0/0.5两臂、同checkpoint/beam20/最终dense排序，复用trace和配对汇总。
- 冻结新增两次手动推理预算、累计预算与停止规则，提供测试和命令。

## Capabilities
### New Capabilities
- `liger-probability-mixture`: 条件概率混合候选生成和匹配对照。
### Modified Capabilities
无。

## Impact
LIGER logits processor、模型配置与trace元数据、聚焦测试；不改训练、不引入依赖、不读取归档效果。
