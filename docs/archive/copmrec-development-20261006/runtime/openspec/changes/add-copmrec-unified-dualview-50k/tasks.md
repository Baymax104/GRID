## 1. 证据与预算

- [x] 1.1 核真实v5训练/Val/CI/bucket/trajectory，保留部分正向、原gate false及源。
- [x] 1.2 固定唯一双目录评分、预测/反证及原150k内第2槽用途；披露双seed尚无法在余1槽完成。

## 2. 实现与检查

- [x] 2.1 新v5.1薄类，单projection/单query/单matmul/平均后不再normalize，严格CP scoring契约。
- [x] 2.2 薄model/experiment/根脚本；继承完整50k配置，notes/dry-run/override/quoting/Hydra验证。
- [x] 2.3 聚焦等价/梯度/cold/归一化/CP测试与独立review，旧390源字节保持。
- [x] 2.4 新来源/driver/auditor绑定、官方sync、实际CPU及双卡一步smoke。

## 3. 唯一新验证

- [x] 3.1 v5.1 seed42从0完整50k，审计实际预算、rawbest、fullstate/source/输入。
- [x] 3.2 ownbest单卡完整Val，独立raw对native42和冻结v5，判定双8及覆盖/头部预测，不运行Test。
- [x] 3.3 记录所有成本与真实正负/不确定结果，修订余1槽用途的决定，保持goal未达时active。
- [x] 3.4 实际证据独立review及OpenSpec严格validate；不将局部方案完成等于双seed目标完成。
