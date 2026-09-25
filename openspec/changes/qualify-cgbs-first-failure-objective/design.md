# 范围和假设

参照 ../../../../research/docs/grid-experiments/2026-09-22-cgbs-objective-redesign.md。只消除监督目标错配这一不确定性，不能证明内容新颖性或 on-policy 有效。

# 冻结规则

- A dragtsrn/39000、既有 50k frontier、branch128、seed42、Adam lr0.0005 / weight_decay0.000001、残差 L2=0.001。
- 每臂 microbatch16、累积2、200次更新；不验证选点、不续训，仅发布第200步。
- 从缓存原始展开长度（第一层 radix、以后原 beam×radix）按生产 topk 找第一次失败；不把补 gold 和 padding 放入选择。
- 首次失败层上，以当前残差重评分后的第 B 个非 gold 合法竞争者作 hinge 边界，margin 固定1.0。A全程存活样本使用最终 A Top-B 内 CE；每窗口一项，平均。margin 是预先冻结的选择，不声称最优。
- 不监督首次失败之后的补 gold CE。两臂保留相同所有层/合法候选残差平方正则（包括补 gold），明确它不是监督收益。
- 记录初始头 SHA、顺序窗口 SHA、曝光数，评估严格核对相等。新目标是固定 A 轨迹近似。
- 评价固定 evaluation 512用户、hash抽样seed20260923；3主条件、1536用户条件，另含复现及缓存重建检查，共4096次beam遍历。
- 有效性先于效果；新目标NDCG必须同时高于A及CE，Recall及前两层存活不低于A。失败则停止；通过仅允许讨论下一阶段，不自动启动。

# 风险

200步只作资格筛选；固定A轨迹与训练后策略有偏差；margin与末层CE尺度不同；已有评价集属于开发集。首次失败和边界训练已有BSO等先例，不能当原创。
