# E1 正式运行结果

用户手动完成 `baymaxam/GRID/dci0bx3o`，2026-09-18 核验为 finished、非 dry-run。冻结协议结论为 `proxy_qualified`。

两次 gain 为 0.054930/0.062064 nats，区间下界均 >0，置乱 p 均为 0.01；90 个合格组覆盖 66.64%/66.98% eval 用户。控制后 Spearman=0.500926，95% 区间 [0.326162, 0.637602]，属于可靠性点估计门槛的边界通过。

输出 `sid-partition-probe-evidence:v0`，digest `798a4978bf039d9db78f0079dee67e9d`。8 文件 manifest、三个实现指纹和导出数据的关键统计复核通过。原始 training 文件及输入 tensor 未独立重读；未重跑实验。

完整身份、统计、局限、路线判断与证据见 [研究结果报告](../../../../research/docs/2026-09-18-sid-partition-dci0bx3o-results.md)。E1 完成，只进入 E2 协议审查；未建立量化因果瓶颈、新颖性或推荐效果，E2/E3 未实施、未运行。
