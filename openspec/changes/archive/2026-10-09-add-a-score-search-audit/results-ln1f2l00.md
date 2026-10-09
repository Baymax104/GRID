# G0 审计结果：ln1f2l00

## 决策

建议进入 G1 的有界评分优化探针：同起点 CE 续训与 CE＋完整 SID 排序对照。本次没有观察到精确搜索带来的新增目标命中，不优先扩展 beam 或改写搜索。该判断仅用于开发顺序，不证明排序训练有效，也不否定其他用户上的搜索收益。G1 尚未实施或启动。

## 来源与完整性

- [W&B run](https://wandb.ai/baymaxam/GRID/runs/ln1f2l00)：finished，runtime 121 秒，Beauty/evaluation，seed42，物理 GPU0。
- 原 A：`5g3wpbg7` 的 19000 步 checkpoint；实际 used Artifact 为 `tiger_catalog_grounded_token_content_init_beauty_train-checkpoint:v0`，digest `aaba5b47dc9a5e94c0d23b75cf697238`。
- SID：`rkmeans_inference-semantic-id:v2`，digest `20f08b323a286fbb3f16b5ea27562af1`；embedding：`sem_embeds_inference-semantic-embedding:v5`，digest `ab56af975eac589c27eed6094482cbd7`。三条上游 lineage 均存在。
- Evidence：`baymaxam/GRID/tiger_a_beauty_score_search_audit_seed42_n128-evidence:v0`，digest `5b124f5356bd0dd144fb7f5725219e1a`，单文件 `a_score_search_audit.pt`，6,476,995 bytes。
- 128 个唯一用户与 requested_users 一致，目录 12,101 items，beam10、chunk128、sampling seed20260918，highest FP32。调用独立 validator 重算排名、TopK、分类、前缀生存及概率一致性通过。此处未重新遍历原始 evaluation 数据重建抽样集合。
- 本地复算：`tmp/a_audit_ln1f2l00/analyze.py`；完整 API 快照与统计为同目录 `run.json`、`analysis.json`。

## 配对结果

| 指标 | 实际 beam10 | 全目录精确 Top10 |
|---|---:|---:|
| 目标命中人数 | 8 / 128 | 8 / 128 |
| Hit@10 | 0.062500 | 0.062500 |
| NDCG@10 | 0.04189560 | 0.04155322 |
| 独有命中 | 0 | 0 |

全部 120 个 beam 未命中均为确定评分失败：目标精确排名下界大于10。确定搜索遗漏0、跨Top10边界不确定0。精确NDCG略低，差值为−0.00034237：user18799的目标在beam排第4、精确排第5；其余7名命中用户的名次相同。精确概率排序不是推荐相关性的理论上界。

平均 Top10 集合重合率为68.4375%，只有14/128用户的两个集合完全一致。因此“本样本无新增命中”不等于“beam已精确恢复目录Top10”。

目标精确 rank≤10/20/50/100/500/1000的人数分别为8/13/23/33/59/67。实际beam目标前缀在四层后的生存人数为49/18/11/8；首次失败人数为79/31/7/3。较早失活是观测现象，不能由此推断改输入结构或首层损失就会有效。

## 数值与成本

本次所有用户概率质量最大误差为2.04e−7，beam/teacher logp最大差为1.43e−6，低于1e−4检查阈值；highest修正在本次真实GPU运行通过。但没有失败run的逐项误差，不能声称已受控证明medium是唯一原因。

encoder总耗时0.937秒，目录评分82.521秒，beam3.252秒；目录评分/beam部分耗时约25.38倍。这是本次batch1、highest、同一编码复用条件下的测量，不是历史部署或端到端吞吐倍数。

## 下一步与结论边界

沿用既定 G1：同一A权重起点，两个分支均重新初始化Adam，lr5e−5，各2000更新，每500验证；CE对照与CE＋lambda0.1/K2完整SID排序。必须同时比较纯CE续训和冻结A，并检查Recall、末三点稳定性以及训练成本。详细门槛见design.md。

本审计只有一个checkpoint、128个evaluation用户、8次命中，未触及testing，不用于宣称统计显著或新方法增益。评分失败占多数只说明精确搜索没有救回这些目标，不证明特定优化目标可改善它们。先做两分支小探针，不扩展seed或数据集矩阵。
