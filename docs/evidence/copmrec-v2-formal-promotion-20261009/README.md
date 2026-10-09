# CoPMRec v2 正式设计冻结与实现回执

2026-10-09用户选择v2：在v1上删除native view/native CE，SID CE、joint CE、mixture NLL保持，训练/Validation/推理均不排除历史商品。正式定义已确定；本次0新训练/0正式Testing/0diagnosis，无GPU dry-run。

- [正式定义与选型边界](../../../../research/docs/copmrec-v2-formal-definition-20261009.md)。
- OpenSpec：`promote-copmrec-joint-only-formal`。
- 新类、组件、两experiment、双卡训练/单卡推理根入口独立于旧v0/v1入口。
- 99项相关用例核验：首次组合96passed/3failed；修复SHA的Hydra字符串引用和归档中旧v1测试名重用判断后，相关15项复跑全部passed。A3有效损失/RNG/全部参数梯度一致；native评分方法不存在；joint CE监督seen目录残差，cold残差梯度为零；严格自身恢复、历史/cold保留、MetricEngine和实际脚本compose核验。
- 原W&B run、Artifact、checkpoint、已完成issue和旧指标不追改。A3仅为选型依据，正式新矩阵未运行。
- [验证记录](verification.json)、[Linear同步前快照](linear-before.json)、[Linear最终回读](linear-final-readback.json)。
