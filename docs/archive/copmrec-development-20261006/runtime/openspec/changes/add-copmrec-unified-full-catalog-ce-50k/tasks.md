## 1. 问题与最后槽

- [x] 1.1 汇总v5正向、v5.1否证、真实cold现有输出及训练支持集差异，固定主／辅预测与因果边界。
- [x] 1.2 登记唯一v5派生干预、原150k内最后50k及完整Val、负／不确定停止；保留整体复现缺口。

## 2. 实现与运行准备

- [x] 2.1 新v5.2类：训练全目录CE、原loss／dropout顺序、严格dense_ce_support，旧396运行字节保持。
- [x] 2.2 薄model／experiment／根脚本与聚焦CPU、Hydra、Bash参数验证及独立review。
- [x] 2.3 新driver／只读auditor／来源绑定；累计账本不重置，未分配Test／43 fail closed。
- [x] 2.4 官方同步、实际随机起点CPU及DDP2一步smoke通过，唯一正式启动。

## 3. 实际最后验证

- [x] 3.1 真实连续50k／100raw点／ownbest／fullstate/source／输入独立审计。
- [x] 3.2 ownbest单卡完整Val，独立全量raw／paired对native和冻结v5；报告cold风险及主／辅预测，不Test。
- [x] 3.3 完整累计成本与原双8判定、保持未达goal active；不自动扩预算／换模块。
- [x] 3.4 实际证据独立复核及OpenSpec严格验证，不将单seed方案结果等同整个目标完成。
