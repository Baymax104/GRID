# CoPMRec v5.2：完整目录content CE的50k鉴别

## 问题、依据与当前授权

原150k累计预算中的最后一个50000训练槽及最后一项完整Validation已由唯一v5派生模型v5.2完成。模型随机初始化、全部模块从step0共同训练，单次连续50000更新，单checkpoint部署。训练run `rha8mrvs`已finished／exit0，主审计与独立终态复核通过；完整Validation run `d7lcftto`已finished／exit0，175原始文件、22363用户的独立原始输出审计通过。相对同history政策native42，R10+4.658672%、N10+12.968595%，两paired绝对CI下界均正，保留v5.2的部分正向结果；R10仍低于+8%，整体门禁false、goal仍active。尚无seed43成对复现或新Testing。用户授权的多卡训练／单卡推理没有扩大原累计上限。

## 实际结果与当前决定

以下均来自[完整Validation审计](evidence/copmrec-unified-full-catalog-ce-50k-20261006/inference-val-candidate42.json)，不是训练期raw dense曲线或W&B summary替代值。

| 指标 | v5.2完整Validation | 相对同政策native42 | paired绝对差CI95 | 相对冻结v5 | paired绝对差CI95 |
|---|---:|---:|---|---:|---|
| R10 | 0.10146223673031346 | +4.6586715867% | [0.0007590663148951393, 0.008227876402987076] | +1.0240427427% | [−0.0008496176720475786, 0.0029077046907838777] |
| N10 | 0.061058281374572636 | +12.9685951139% | [0.004809319752127361, 0.009381575180557149] | +1.5085982519% | [0.0000022124450133101757, 0.0018431268668216742] |

对native新增925／丢失824／净101，共同命中上移577／下移413；对冻结v5新增237／丢失214／净23，共同命中上移532／下移461。v5增量的R10 CI跨0、N10 CI下界仅略高于0，且两点增量均未3%，不能宣称完整目录CE已经带来明确物质增益或证明cold占位导致此前Recall损失。当前保留依据是整体v5.2相对native的部分正向证据，不把原共享残差与本次CE干预的作用混为一项已证明机制。

[固定双8门禁](evidence/copmrec-unified-full-catalog-ce-50k-20261006/candidate42-validation-gate.json)要求R10≥0.10470151589679383及N10≥0.0583728104417629。实际N10通过、R10未通过，不进入原晋级后的seed43／Testing，不宣布整个goal完成。附加[独立推理复核](evidence/copmrec-unified-full-catalog-ce-50k-20261006/validation-independent-review.json)与一次[预固定cold只读检查](evidence/copmrec-unified-full-catalog-ce-50k-20261006/bounded-cold-result-review.json)已经完成。false cold槽6638→44，已命中seen目标前cold槽209→0；相同2032共同seen命中用户的该计数175→0，51个cold-target用户的正确命中6→0。辅助点方向得到支持，Recall干预增量仍未确认，cold真命中取舍完整保留。单seed、重复使用开发集、固定checkpoint条件下的逐点CI及历史native训练source缺口继续披露。

保留v5已确认的NDCG／Top5收益。其完整Validation相对native42为R10+3.597786%且CI跨0、N10+11.289681%且CI为正。v5.1固定双目录平均相对v5的R10−5.031167%、N10−7.157376%，两CI均负；该具体干预已停止，不扫描权重。[v5.1五项路线门禁](copmrec-v5-1-stage-decision-20261006.md)

现有三套输出的只读统计重核了PT／CP字节、共同目录和175原始文件／22363用户，未生成新分数或推荐列表：

| 方法 | cold Top10槽 | false cold槽 | 曝光用户 | 已命中seen目标前的cold槽／用户 |
|---|---:|---:|---:|---:|
| native42 | 216 | 216 | 216 | 11／11 |
| v5 | 6644 | 6638 | 3657 | 209／116 |
| v5.1 | 11113 | 11105 | 7375 | 340／257 |

v5新增／丢失seen-target case的cold存在率17.09%／18.91%，两个固定key组方向不同，关联较弱。v5.1相对v5的489个seen-target丢失中，309个没有cold槽。因此，cold不是已被证明的Recall单一瓶颈；多数错误槽变化是错误商品之间的替换，没有Top11不能估算可恢复命中。[实际只读分析](evidence/copmrec-unified-dualview-50k-20261005/cold-occupancy-existing-output-analysis.json)

当前可执行训练的具体差异是：`JointMixtureLiger._joint_losses`的training content CE将cold原logits替换为−100，使这些原logits没有直接的content CE梯度；mixture仍读取未mask的完整logits，部署也允许cold。这是GRID baseline共用的既有行为，不能称为官方bug，也不能说cold完全没有监督。v5.2仅检验补上这部分直接竞争监督是否有推荐价值。

## 固定方法与相关工作定位

采用 `UnifiedFullCatalogCECoPMRec(UnifiedScratchCoPMRec)`，版本v5.2。保持v5单history query、共享seen residual／cold残差零、单目录`normalize(p_i+r_i)`评分、三loss各1及learned alpha。训练content CE改为完整目录softmax；SID CE与legal-prefix mixture NLL沿用原式：

```text
content_CE = cross_entropy(full_catalog_logits, seen_training_target_row)
total_loss = SID_CE + content_CE + mixture_NLL
```

不加入history CE mask、第二query、teacher、cold item bias、先验阈值或额外loss权重。cold仍有推理资格；训练目标仍必须是seen商品，cold residual仍为零。新梯度会改变query、projection与seen residual，不声称query或其他已训练参数保持不变。

[LIGER原论文](https://arxiv.org/html/2411.18814v2)第3节已有内容CE与生成CE联合建模，以及加入cold商品后用内容表示排序；本文干预定位为保留协同残差后对训练竞争支持集的具体鉴别，不将普通完整目录softmax宣称为首创。论文公式的集合符号不足以证明官方实现是否mask cold；本次行为判断绑定GRID实际代码。

固定支持集字典由模型契约、experiment顶层和writer metadata一致记录：

```yaml
dense_ce_support:
  protocol: copmrec-all-catalog-dense-ce-v1
  training_support: all_catalog
  cold_items_in_denominator: true
  training_targets_must_be_seen: true
  validation_support: all_catalog
```

父`unified_scratch_contract`版本为v5.2并增加上述字典，拒绝错误支持集、v5／v5.1或其它模型checkpoint。优先只新增文件，保留此前396个运行文件实际字节；保留原loss计算顺序与dropout调用顺序，不通过提前目录投影改变随机数使用。实现不增加参数，需如实记录实际计算变化，不能据参数相等宣称相同FLOPs。

## 唯一训练与评价协议

- seed42推荐模型随机初始化，gate/residual零，0→50000连续更新，无外部推荐checkpoint、warm-start、optimizer重启或pool。
- AdamW：主干peak0.0003、residual0.002、WD0.035；warmup2500、cosine horizon50000、min ratio0；FP32、clip1。
- DDP2每卡128、global256、accumulation1；同两个SID/content Artifact、原causal max32／有效history20／sequence80。
- 每500更新自己的raw dense Val NDCG@10选best，保留first最大值tie规则；完整50k预算与已保存best/last状态分别核验。
- ownbest仅一次单物理GPU、local[0]、单进程完整Validation，原history排除、stable catalog row ties、全部cold资格。
- native42 `wdms8w77`及其真实50k训练 `35ig0tz6`是固定主对照；冻结v5 `8w893ra3`用于本次增量，v5.1负结果与cold统计用于解释。不挑CP、alpha、温度、loss权重或seed。

整体主要门禁原样保持：R10≥0.10470151589679383、N10≥0.0583728104417629，即相对同policy native各至少+8%，两个paired绝对差CI95下界均为正。Bootstrap仍为PCG64 seed42／2000次，同用户配对。原始用户／标签／输入／CP／source／合法唯一零history审计必须先通过。Beauty Validation既有开发使用及native42历史source缺口继续披露，不由新归档回填。

## 预测、风险与结果触发的决定

主要预测是相对保留v5提高R10并保留NDCG优势；辅助预测是false cold槽及already-hit seen目标前cold槽减少。辅助统计只作机制描述，只减少cold槽而R10不改善，不算解决推荐问题。

训练没有cold正目标；加入直接负例竞争可能降低cold推荐。完整报告51个cold-target用户的命中和整体指标，不排除cold用户获得提升，不声称小组冷启动泛化。

若通过原native双8%与双CI门禁，冻结这一单seed合格模型，明确第二seed／Testing尚未完成；v5增量与辅助预测分别评价，不增加一个任意的v5非劣门槛替代用户主要目标。若R10相对v5不正，覆盖预测反驳；若N10对v5的CI全负，保留原排名的预测反驳，即使主要目标合格也须如实报告取舍，不维持该机制叙事。若未合格但有≥3%明确增量且另一指标无点退化，按CI强度保留部分证据；CI跨0只称不确定。没有明确正向价值时停止本次CE-support路线。无论结果如何，不扫描loss权重、温度、cold规则或转接双query维持同一假设。

## 累计成本与实现门禁

原上限3train／150000更新、3完整Val、3新Test保持且没有重置。训练已启动并完成3／3，实际核验更新及已承诺更新均为150000；完整Val已启动并完成3／3，无in-progress或reserved项。正式Validation启动attempt4包括1次保留的预测前失败；新Test实际0、当前计划0。没有未分配训练槽或完整Val槽；seed43成对两模型未分配，复现目标保持，不能把本次单seed部分正向结果改称整个goal完成。后续只读分析不增加模型训练或正式评价次数。

先完成新模型、薄配置／根脚本、聚焦gradient／eval等价／CP契约测试、Hydra与shell参数检查、来源／auditor绑定和OpenSpec strict；随后官方Mutagen flush/status、实际CPU随机起点预检与双卡一步smoke通过，才唯一启动正式训练。完整Val在实际训练审计通过后做strict CPU恢复，保留唯一job/PID并按终态观察，不因轮询超时重启。

## 实际启动与当前证据边界

- 训练唯一PID `766535`，job `logs/autonomous/copmrec_unified_full_catalog_ce50k_train_candidate42_20261005T190422711184Z`；物理GPU6、7→`CUDA_VISIBLE_DEVICES=6,7`→本地`[0,1]`。冻结终态observation证明PID已退出、exit0／finished；100个500步间隔raw Val事件、summary step49999与终止日志共同证明实际50000更新。主审计和独立终态复核已完成。
- own raw dense Val N10首个最大值选中实际47000 checkpoint，URI `wandb://baymaxam/GRID/rha8mrvs?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=047000.ckpt`，SHA `a82c2dc1cf7857d92c90915bd841193de9ed6af9e17394f03d624d18171c764c`。best与saved last均为47000，165 state／154 optimizer moments及连续scheduler只核验到47000；没有50000终态完整checkpoint。实际训练预算50000与保存状态47000分别披露，last没有替代best或被重标50k。
- 实际source402文件，SHA `16142f5a8375936b3be40f87f4fb8012041ce9427a148b7a6ad0e1e49e23b9e0`；此前396文件字节全部保持。主审计已核实际运行archive／manifest全字节，额外source独立复核已通过；后者不冒称重新远端读取全部checkpoint状态。早先service partial记录保留原边界，由单独终态审计补齐，未覆写成成功证据。
- 唯一完整Val PID `2995739`，job `logs/autonomous/copmrec_unified_full_catalog_ce50k_val_candidate42_20261006T053759565028Z`，物理GPU7→本地`[0]`／单进程。实际strict CPU恢复先通过（0forward／0Trainer），随后正式run `d7lcftto`完成；bundle的SID合法唯一、零有效history、真实raw标签与输入／checkpoint／402 source均独立核验。
- 聚焦核心最初新25＋旧45／27共97项通过；strict-bool恢复修正后新33项再次通过，未重跑未改变的旧72项。配置40项、编排59项拒绝检查与实际2个shell／Hydra capture、训练／推理auditor纯自检54／86通过，OpenSpec strict通过。
- 实际CPU起点参数11031809、165 state、optimizer空、双groups `.0003/.002`、真实20keys支持集合同；无外部推荐CP、forward或正式run。官方同步flush/status成功、三session Watching无冲突；实际DDP2一步smoke rank0／1、exit0、W&B0，绑定同source。
- 类型检查在新类和新docs契约路径内严格拒绝`True == 1`穿透。预启动仅本地编排guard修正已由独立fixture验证，未改旧runtime或启动其它正式run；原始实际准备、CPU、smoke与实施凭据均保留。

[累计账本](evidence/copmrec-unified-full-catalog-ce-50k-20261006/cumulative-budget-latest.json)、[实际终态与Validation登记](evidence/copmrec-unified-full-catalog-ce-50k-20261006/training-completion-and-validation-registration.json)、[训练主审计](evidence/copmrec-unified-full-catalog-ce-50k-20261006/training-candidate42.json)、[训练独立复核](evidence/copmrec-unified-full-catalog-ce-50k-20261006/training-independent-review.json)、[source独立复核](evidence/copmrec-unified-full-catalog-ce-50k-20261006/source-service-independent-review.json)、[实施凭据](evidence/copmrec-unified-full-catalog-ce-50k-20261006/implementation-verification.json)。登记前ledger及research-state原字节均单独归档，旧closed tail SHA保持 `1b72d753b98633cbd6b9da38e3d668b25ef767d9056127494564b1f6acf3bee8`。当前保留部分正向结果，整体双8与复现尚未完成，未做新Test／seed43；不追加或重置预算。

## 阶段结题与下一额度

[五项路线决定](copmrec-v5-2-stage-decision-20261006.md)已经记录：保留v5.2部分收益，停止继续压cold占位，原3train／150k及3完整Val封口，整体双8和配对复现目标尚未完成。[下一固定辅助视图CE方案](copmrec-v5-3-native-view-ce-proposal-20261006.md)仅形成可审阅proposal，尚未实施或分配新增额度，既有预算不重置。
