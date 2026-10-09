# DASFAA 当前执行看板

2026-10-06：用户确定 **CoPMRec v5.3** 为正式版本；2026-10-07 明确 **LIGER hybrid 为正式基线，LIGER dense 仅作内部对照**。2026-10-08正式主矩阵9/9核验完成，原LIGER9组正式结果保持复用；BMX-117的有界五臂消融和三个机制证据包现已完整核验交付。所有开发结果仅归历史，旧看板原字节已保存至 GRID `docs/archive/copmrec-development-20261006/active-document-before-reset/research-dasfaa-2027-execution-board.md`。

## 当前工作包

| 工作包 | 本轮安排 | 状态 |
|---|---|---|
| M1 协议与上游 | 保留三数据集冻结技术输入，更新正式评分与来源协议 | 输入 digest / 执行前审计待登记 |
| M2 CoPMRec 主结果 | Beauty / Sports / Toys × seeds42/200/2026；9训练、训练内选点、9Testing及独立复算 | 9 / 9完成，BMX-116闭合 |
| M3 消融与机制 | Beauty/seed42，A1～A5共5训练/250k/0额外Val/5Testing；M1 bundle、M2残差2×2、M3 Full/A1/A4前缀概率 | 5/5训练和5/5Testing核验完成；M1零forward、M2已交付、M3三来源完成；4/4 diagnosis，工程attempt5含1失败 |
| M4 LIGER hybrid | 复用原9组正式训练、own-best及Testing，保留原生hybrid和历史资格协议 | 9 / 9原Done/有效保持，配对输出已独立核对 |
| 内部 LIGER dense | 复用同一既有LIGER own-best，不另训练、不增加Val、不替代正式 baseline | 单列9完整Testing，internal阶段；未纳入本次BMX-117授权 |
| M4 其他基线 | TIGER / LETTER / SASRec 既有有效状态保留，进入主表前审查共同评价协议 | 按各 issue 状态，不自动重置 |
| M5 论文 | 主方法为v5.3；正文与主表引用正式主矩阵，方法消融与机制观测引用新M3证据 | 主矩阵和M3实证已具备，正文按原值/差值/区间与解释边界组织 |

## 执行顺序与边界

1. 启动前核对实际输入身份、raw split、代码与同步状态；保留运行时源码字节、manifest/digest和物理GPU到local设备映射。
2. 按用户2026-10-08明确授权，BMX-122/123/129/142/143分别在node1独立tmux启动双卡scratch50k训练；启动后登记W&B run ID，当前无需持续监控。
3. 训练完成后审计各自训练内raw val/ndcg@10首次best，再做一次单卡Testing和独立指标复算；不增加独立Validation，不复用Full或旧开发checkpoint初始化。
4. BMX-145固定Full残差2×2已交付；BMX-146 Full/A1/A4各自own-best的三源前缀记录完成；BMX-144用Full+五臂已核验bundle完成零forward分析。来源均真实核验，未用Full替代变体来源。
5. M3只报告原始观测、带符号差、N和区间，不给好坏/积极/消极标签，不据方向扩预算；Done是完整交付门禁，W&B finished不单独构成Done。

完成的主配对成本为9个新CoPMRec训练/450k更新/0额外独立Validation/9Testing，LIGER原9组正式结果复用。BMX-117独立预算已执行完成：**5训练/250k实际更新/0额外独立Validation/5Testing**；另**4/4 diagnosis任务/6新增全量forward等价pass**，M2 V11复现与失败重跑作为工程核验开销单列。工程attempt5含`ls14h0dw`序列化失败，正式完成只计4；M1 `m3hitc8p`为六份固定bundle的另一次分析、0 model-forward。M2 `717vgnkn`、Full/A1/A4 prefix `j2rworuj/5murlou7/pltnvpjf`均通过独立统计与Artifact文件身份核验。内部dense的9Testing仍另计、未纳入本次授权。

五训练保持原始source SHA`42f0d7c7dce0a09961e0c8c23f9ba35982abfb796d13952c3f17eaeba17c9f2e`，Testing/diagnosis为`5e95cf7e87b28b730976b5a6d279acdc7aca3a270a08dd5e4519ce1fc6624867`，两处metadata序列化变化分别登记，不回填训练来源。全部N=22363、同user-label SHA，Top10合法唯一、history exclusion/cold资格与四指标复算通过；M1为114 slice/608 CI，含A4−A1和A5−A3及K5/10三项带符号贡献。原值、完整CI、精确URI/version/digest/MD5/SHA、选点与资源见[最终实证汇总](../../GRID/docs/evidence/copmrec-m3-completion-20261008/m3-empirical-summary.json)、[Testing终验](../../GRID/docs/evidence/copmrec-m3-completion-20261008/testing-audit-summary.json)、[M1终验](../../GRID/docs/evidence/copmrec-m3-completion-20261008/m1-independent-audit.json)与[前缀三源汇总](../../GRID/docs/evidence/copmrec-m3-completion-20261008/prefix-three-source-summary.json)。bootstrap为单seed用户重采样的pointwise区间，teacher forcing不代表自由beam恢复；W&B runtime为原始墙钟观测，active GPU时间与peak VRAM未知。只提供实证，不据方向扩预算。

## 当前权威文档

- [正式版本定义](copmrec-formal-version-20261006.md)。
- [正式实验计划](copmrec-formal-experiment-plan-20261006.md)。
- [当前计划](../ideas/current-plan.md)与[机器状态](../research-state.yaml)。
- [BMX-117 M3统一协议](https://linear.app/baymax104/document/copmrec-v53m3-消融与机制实证协议2026-10-08-643847ec8183)及[GRID实现准备回执](../../GRID/docs/evidence/copmrec-m3-preparation-20261008/README.md)。
- [历史版本与效果](../../GRID/docs/archive/copmrec-development-20261006/versions.md)。
