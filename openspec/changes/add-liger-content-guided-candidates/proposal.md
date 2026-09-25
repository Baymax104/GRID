## Why
当前LIGER同一checkpoint的1146个dense独有命中全部被生成候选排除，且原hybrid有278个独有命中。需一次有边界的干预检验内容提前参与候选保留是否有净收益；无效生成平均6.628/20要求加入合法性对照。

## What Changes
- 增加默认关闭的合法SID约束与基于内容前缀最大分的候选引导。
- 同一checkpoint下仅新增合法性对照与内容引导两次手动推理；保留beam20和最终dense排序。
- 完善配对产物比较、诊断元数据、测试与冻结协议。

## Capabilities
### New Capabilities
- `liger-content-guided-candidates`: 内容引导与合法性对照的生成候选实验。
### Modified Capabilities
无。

## Impact
LIGER生成入口、新logits processor、候选记录汇总与测试；不改训练、数据划分、checkpoint或默认baseline行为，不引入依赖。
