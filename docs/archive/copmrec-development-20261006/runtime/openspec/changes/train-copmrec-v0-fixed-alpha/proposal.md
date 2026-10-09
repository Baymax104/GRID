## Why

v0通过teacher-forcing混合NLL学习全局alpha，真实best48000为0.813255608；v3.1固定推理alpha虽改善高内容候选覆盖，却没有可确认的最终净收益。用户取消尚未启动的v0固定0.813推理，明确改为一次v0固定alpha=0.8从头训练，检验固定权重下的联合表示学习是否改善推荐效果。

## What Changes

- v0新增可选固定训练策略，alpha=0.8从初始化开始冻结；SID CE、content CE与混合NLL三项等权，encoder/decoder/content投影保持可训练，全层Mass保持。
- 训练、验证、推理共享该固定值，checkpoint记录并拒绝混用固定/学习策略，旧默认学习行为保持。
- 薄experiment/model与根训练脚本继承v0，训练随机初始化、无checkpoint，双卡物理2/3、每卡128、累积1、50k更新、dense验证选点。
- 一次用户手动训练额度，无alpha扫描；完整训练agent不启动。取消原未运行推理额度，保留v3.1开发基座和过去结果边界。

## Capabilities

### New Capabilities

- `copmrec-v0-fixed-alpha-training`: 固定权重训练、恢复及匹配v0入口。

### Modified Capabilities

无。

## Impact

仅影响JointMixtureLiger可选alpha策略、新增薄配置/脚本/测试与研究记录，无新依赖。采用已有概率混合与全局gate作为消融，不新增方法主张。最小效果验证：完成50k训练后按原dense val/ndcg@10选点，与7y54j4m6/6dspa7e3匹配比较；后续Testing命令在训练run审核后准备。正向支持该seed/checkpoint设置，负向或不确定保留学习默认，不自动追加训练或扫描。
