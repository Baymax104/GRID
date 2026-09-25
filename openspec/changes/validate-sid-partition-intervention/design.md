## Context

依据 research/docs/2026-09-18-sid-partition-dci0bx3o-results.md 的五项路线门禁，E1 只支持代理资格。历史失败与暂停决定保持有效。ReSID/PRISM/DIGER 等近邻边界沿用已核查记录；本轮是因果探针，不宣称新方法。

## Goals / Non-Goals

实现一次可停止的 E2 准备和三臂匹配训练。非目标：自动运行正式任务、扫参数、新量化网络或直接保证论文新颖性。

## Decisions

### 数据与代理

锁定 E1 输入与记录指纹。E1 folds 0+1 合并 fit，fold 2 为 selection，fold 3 为 internal check。internal check 不参与候选生成/选择；它已被 E1 使用，只是阶段内检查，不是新的独立测试。不访问 evaluation/testing 以选择 SID。沿用 smoothing=20，评价 log(p/q)；同时报告 conditional NLL，防止只通过改变组频次改善 gain。

### 冻结干预 v1

完整四列唯一 SID 按行成对交换，不改 tuple 集合。候选四 item 块来自四个不同首层组，所有 item 在 fit 末目标中至少出现 2 次。seed42 随机选 anchor，在单位内容向量距离最近的 32 个合格 item 中抽取另三项；至多 8192 次尝试、512 个唯一候选块。最近邻只约束候选范围，不使用推荐器结果。

每块索引 a,b,c,d 的引导配对为 (a,b),(c,d)，对照配对为 (a,c),(b,d)。两种配对使完全相同四 item 改变首层，因而任意训练频次口径下改动 item 分布完全相同；每个原组各交换出入一个 item，完整 tuple 集合保证所有前缀目录占用不变。匹配并不保证每组交互频次不变，因此必须报告该残余差异。

几何扰动定义为原 item 单位内容向量与其获得 SID 的原拥有者的 cosine distance；每 item 两臂差异绝对值 ≤0.02，且块均值差 ≤0.01。候选配对和匹配在行为评分前确定。不能使用同一对组内四 item 的两种配对，因为那不会改变首层分组对照。

按单块引导 selection gain 改善降序、候选序号破平局；按此固定顺序一次扫描，接受当前累计映射上仍严格正改善的块，已用 item 不再使用，最多32块。对照使用接受的同一批块的预定替代配对，不根据对照效果选择。少于16块判 insufficient_matched_blocks，禁止放宽门槛。选择阶段不调用 internal check。开发规则只有本配方，不自动重试。

### 准备门槛

检查 tuple 集合/唯一性/首层占用/两臂改动集合与几何匹配。冻结全部映射后才计算 internal check。对 guided-original、guided-matched 的逐用户 gain 差做1000次配对 bootstrap，两者区间下界均 >0，并且 guided conditional NLL 对两者均下降，才判 ready_for_training。否则 no_go_proxy_transfer；这是本干预未晋级，不是所有量化路线无效。dry-run 总是 smoke_only，不输出可训练映射。失败保存诊断但不输出任何三臂 bundle。

### 产物和训练

通过后 shared StructuredAnalysisWriter 原子输出 summary/protocol/identity、候选/接受块/用户/频次 CSV，以及 original.pt/guided.pt/matched.pt 标准 keys+predictions bundle。Summary 保存数组指纹；训练 guard 校验门禁、arm、映射与 embedding 指纹和训练数据目录。W&B 使用单一 sid_partition_intervention artifact，所有输入通过共享 resolver，callback 记录 lineage。

三臂使用现有 TigerCatalogGrounded 的 mask_ce（随机 token 初始化、合法前缀 CE），同一 seed42、20k optimizer 更新、每500 batch验证（累积=1时即500更新）、best val/ndcg@10、不训练后自动 testing。同一硬件/批量/精度/数据配置，分别从头训练。全量 evaluation 预测及128用户精确审计使用现有 pipeline，并绑定各自 SID 和 checkpoint；不拿旧 checkpoint 套新 SID。

### 预算和判定

本阶段增加1个CPU准备 run（可先单独 smoke），成功后最多3训练、3 evaluation预测、3次128用户精确审计。E2 相对原始/匹配对照 NDCG均至少+1%、Recall不降，报告配对区间、代理与item方向、成本。否则停止。只有E2通过才考虑原冻结E3三个seed43训练；本轮不实现自动矩阵。最多六训练/120k更新不变。

## Risks / Trade-offs

- 改动64–128 item可能不足以影响总体排序 → 记录覆盖与代价，失败后不自动扩大预算。
- 代理选择过拟合与组频次重分配 → internal check只读一次，配对区间与conditional NLL双检查，并导出各组频次变化。
- 相同改动item与几何约束仍不构成随机因果识别 → 只解释该受控干预，配对结构与组交互频次残余差异公开。
- 非独立internal check、单数据/seed与最近邻重叠 → 不据准备或单臂增益宣称论文完成。
- 匹配可能不可行 → 记录停止证据，不调容差重试。
