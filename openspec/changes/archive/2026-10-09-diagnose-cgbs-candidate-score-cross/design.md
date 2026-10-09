# 设计

复用原encoder和decoder，逐用户两组Top10的并集最多20条SID，decoder前向同时产生原始token与原CGBS逐层混合评分。合法归一化、alpha、目录原型与原模型一致。每条完整路径累加四层log概率。批次<=8、单进程、FP32、evaluation、原content_init_full。

输入经过artifacts共享加载，保留W&B lineage；两组trace须有相同checkpoint_reference/catalog契约与用户标签，模式分别trained/off，候选唯一合法，原trace末层目标rank与候选匹配。新数据每批标签再次比对。

分数按降序排序，同分容差1e-5以内分组保留输入候选顺序；对角格须逐用户重现原次序，否则失败。容差组最大最小差不得超过阈值。共享writer保存scores/order/target_rank/ndcg，保留键与标签。完整运行用户集合必须等于输入集合；dry-run仅有界烟测，不发布完整证据。

最先运行CPU微型原beam对比重评分测试，真实GPU由用户手动启动。结果为有限Top10候选对照，不是全目录反事实，不用testing调参。

数值复现修复：评分沿用统一入口matmul精度medium，不强制切换highest；输出记录实际精度。固定1e-5门槛不变。旧版第6批用户15785的on第8/9名交换，修正后同48用户通过。

后续 user 9391 仍有 1.66893005e-5 逆序。完整 teacher forcing 改为逐层 BOS+prefix 的 decoder 前向和末位置 lm_head 投影，与 beam 的序列长度一致；metadata 标记 path_scoring=prefix_replay_v1。沿用原精度与容差。显式故障回放保留指定用户所在的原始八人 batch，要求 workers=0，关闭 writer；不用于完整结果发布。候选并集的 decoder batch 与历史 beam 仍可能不同，有界检查不能保证全量数值排序复现。

user 15864 的 off 前两名仍逆序4.14848328e-5，单个原八人 batch 复现到逐值一致。增加显式 reproduction_audit=false（默认关闭）开关；开启时只延后对角失败判定，其他校验即时执行。共享 writer 的领域子类收集失败用户、分数、候选、输入指纹、原目标排名和重评分排名及 NDCG 差，保存 reproduction_audit.json。完整覆盖且零失败才报告 passed；任何不完整或逆序在写报告后抛错。该模式始终不写正常 candidate_cross.pt 或发布正式 Artifact，不用于机制结论。不更改1e-5门槛，不把小分差自动归类为数值误差。
