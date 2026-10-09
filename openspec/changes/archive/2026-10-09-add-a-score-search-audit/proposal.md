## Why

二层内容交互未通过开发门槛，下一研究方向是保留A的完整SID排序训练。此前MIR/BRIR审计不能判断A的失败是评分不足还是beam遗漏，需要先完成A自己的有界冻结审计。

## What Changes

- 增加原A专用的全目录精确路径概率与真实beam10比较，通过统一预测入口、共享writer发布证据。
- 复用确定性用户抽样，不采集新的testing结果，不训练模型；保存完整概率、目标排名、生存、数值边界和身份。
- 冻结G1两臂各2000步续训协议，只有G0支持后才另行实施；本变更只实施G0。

## Capabilities

### New Capabilities
- `a-score-search-audit`: A冻结checkpoint的评分与搜索分离审计。

### Modified Capabilities
无。

## Impact

新增catalog-grounded审计子类、数据证据验证、薄配置、单卡根脚本与CPU测试。复用已有抽样datamodule和AuxiliaryTensorWriter，不修改A训练/生成行为，不新增依赖。
