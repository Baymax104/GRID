## Context

v4.3历史资格层已经保留：自身R10+8.568824065633551% / N10+17.367341655669733%，188新命中、0丢失，双CI正；对same-policy LIGER为+9.870848708487067% / +10.28369953005015%，整体Recall门槛未过、未Testing。新投入依据这个已确认的问题与源码支持集差异，不能仅因少3个命中扩预算。旧5训练/30000封存，root在用户自主方法/SSH授权及最新组件保留指示下明确新加两臂各2000，总限7/34000；旧结题不改。

实际 `joint_mixture._joint_losses` 仅训练dense CE对cold设-100，随后原raw logits仍传给合法mass-prefix mixture NLL。当前有效history没有从该CE竞争集合排除。我们只检验训练content CE支持集与已确认推理资格的局部对齐，原策略是否有害、修改能否改善推荐都未知；bad case的训练频次及未投影content cosine不是根因证明。

## Goals / Non-Goals

**Goals:** 两臂从同一个v4.1冻结bias0 checkpoint weights-only初始化，仅训练content CE历史支持排除不同；保留训练与部署的有效history资格、严格来源与公平baseline，以完整有界评价判断history-CE增量、单纯续训增量和整体目标。

**Non-Goals:** 改SID/mixture支持集、三loss权重或learned alpha；新增bias参数、CF索引/打分、temperature/alpha/LR/窗口/成员/权重扫描；用loss下降声称推荐收益；用Testing选择checkpoint/臂；恢复旧mixed/bias/CF结论或重置预算。

## Decisions

1. 新 `src.recommendation.liger.eligible_content_continuation.EligibleContentContinuationCoPMRec` 继承 `ItemScoreBiasCoPMRec`，真实version `v4.4`。原kwargs外增加严格bool `exclude_history_from_dense_ce`（treated=true/control=false）、`pretrained_v4_1_checkpoint`（公共loader dict或null）、必需 `expected_continuation_reference` / `expected_continuation_sha256`。旧v0/v4预训练字段均null，bias_scale0且参数保持0、residual_scale1 / multiplier20、dense/content、关闭candidate/path trace；两臂没有其他结构差别。
2. 每个训练batch只进行原encoder/query/raw dense logits计算一次。原SID CE与mixture NLL沿用原函数、原raw logits；control直接沿用原三loss，treated仅替换content CE：先按原cold=-100，再将当前输入attention有效完整SID行设为-inf。保留原三loss权重各1，最终 `loss=sid_loss+content_loss+mixture_loss`。排除只应用 `training=True` 的content CE，validation loss定义仍原样；两臂用于checkpoint选择的推荐指标均由history-excluded dense排序产生。
3. 输入history固定与v4.3相同：最多20个有效完整SID、80tokens/4hierarchy、右侧padding、attention0/1，inactive槽值忽略，精确catalog lookup与去重；未知/partial SID失败。训练及Validation target必须不属于有效causal history，检测只能验证数据协议，不用于改变mask或把target“放回”。完整训练raw唯一性/实际causal preprocessing须先核验，运行期按batch拒绝重叠；若不满足则停止准备并修正数据假设，不能静默过滤样本。推理helper仍不读labels/raw training/user-key。
4. CE必须保持autograd：被排除history行对该CE直接梯度0，未排除行保留正常梯度，loss和所有梯度有限。既有 `apply_history_exclusion` 带 `@torch.no_grad`，不能直接把其返回值作为训练logits；实现共享相同SID/attention资格语义或无梯度rowmask，然后对原梯度tensor做masked_fill。检查control与原三loss逐值/梯度相等、同权重下treated的SID/mix逐值相同，以及支持mask/梯度；这些只是实现验证，不证明训练轨迹中的SID/mix数值不变。
5. 唯一warmstart为 `wandb://baymaxam/GRID/l3zyr91b?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt`，SHA `4bea4d5771c4f056ad6e0cd69d1204a3220b639868803cbc19aa3332ecf0a3c9`。用原native v4.1 validator正常hook/strict state核对原文件，保留catalog、bias0、残差与原v0/v4来源链，再严格复制全部权重到新模型；不修改原checkpoint版本来伪装验证。辅助native构造用fork_rng，不扰动两臂相同RNG起点。顶层ckpt_path=null，optimizer/scheduler/global_step均从0新建。
6. 新checkpoint保留原 `copmrec_pretraining` 和 `copmrec_score_bias_pretraining`，另增真实 `copmrec_continuation_pretraining={version:v4.1,global_step:6000,weights_only:true,reference,sha256}` 及下面exact9训练contract，对应property `continuation_metadata` / `eligible_content_contract`。normal v4.4恢复严查flag/来源/原父契约和statekeys、合法int step0..2000并记录restored_step，允许smoke正常恢复；仅正式pool部署要求selected best1000或2000。不把原v4.1/6000身份当成新保存checkpoint的版本或step。

```yaml
copmrec_content_eligibility:
  protocol: copmrec-dense-ce-input-history-v1
  exclude_history_from_dense_ce: true # control为false，必须与真实checkpoint一致
  history_scope: input_valid_complete_sids_max20
  applied_to: training_content_ce_only
  cold_policy: seen_mask_minus100
  sid_loss: unchanged_full_vocabulary_ce
  mixture_loss: unchanged_full_catalog_legal_conditionals
  loss_weights: [1, 1, 1]
  alpha_policy: learned
```

7. 两臂固定AdamW两组：主干LR.0001、item/residual LR.002、WD.035；bias冻结且不进optimizer，cold residual0不变。FP32、seed42、GPU2,4→local0,1、每卡batch128/global256、accumulation1、原clip和数据顺序保持。fresh scheduler warmup300，**scheduler_steps显式6000**，min_ratio0，max_steps2000不缩短cosine horizon；每1000步validation，完整2000步终态，两次history-excluded dense N10选best1000或2000，同时保留固定2000步标量对比，不使用Testing选择。原配置插值 `${trainer.root.max_steps}` 不能直接继承为2000。
8. 新 `HistoryExcludedContinuationPool` 与续训类同处新module。constructor仅9kwargs：catalog、source_model_factory、continued_model_factory、source_checkpoint、continued_checkpoint、expected_source_reference、expected_source_sha256、expected_continued_reference、expected_continued_sha256。source仍为nj9elah1 v4 best6000 / SHA `7c52228a665b593b50e45e8aba039f3f4aa4a10ca134d9345c59c59fb142f055`，continued为各臂真实Validation-selected v4.4 best1000或2000。各自由自己原encode/query/catalog raw dense logits产生分数，0.5/0.5平均后应用原exact9 `history_eligibility_contract`，stable catalog-row Top10、cold可选。新pool严格核对continued真版本/实际step/训练flag/continuation来源及原v0/v4链；不能将新checkpoint桥接成旧v4.1/6000以绕过原pool检查。旧pool和旧入口默认契约不松动，顶层pool ckpt_path=null，无wrapper checkpoint。pool_contract保留父公共score/catalog/ties/weights字段，但protocol为`copmrec-fixed-dense-continuation-logit-pool-v1`，control字段替换为continued={reference,sha256,version:v4.4,global_step:actual1000或2000,score_bias_scale:0.0}，另记录continuation_source=continuation_metadata与eligible_content=eligible_content_contract；source原v4/6000身份不变。
9. 新薄model/experiment、根 `copmrec_v4_4_train.sh` / `copmrec_v4_4_inference.sh` 沿统一src.main/Hydra、公共checkpoint/cat loader、lineage/source snapshot/commonwriter；训练mandatory --pretrained-checkpoint映射warm continuation_checkpoint_path/sha256及公共pretrained_v4_1_checkpoint，推理 --source-checkpoint/--continued-checkpoint映射source与新continued_member_checkpoint_path/sha256，warm与新成员字段分开。标准bundle仍keys/predictions、关闭trace；actual metadata记录history原9keys、训练支持contract、真实source/continued身份。单卡完整推理GPU2→local0/NPROC1/FP32/batch32。两新CP的URI/SHA/digest只能由实际selected Artifact写入，不预填旧l3身份。

## Risks / Trade-offs

- 原history负竞争可能有助共享表示，删除会失去有效训练信号 → treated/control匹配两臂，推荐收益而非CE下降决定主张，不预设原CE有害。
- dense CE梯度变化会经共享表示、AdamW、clip影响SID/mix训练轨迹 → 只说修改了content CE支持，不称整mixture对齐或纯最终打分因果；训练后效果依匹配对照。
- 全目录inf或target被mask可能产生NaN → target∉有效history与seen训练target校验，保留eligible数量、有限loss/梯度CPU/DDP smoke；labels只作协议断言。
- weights-only或新pool身份桥接错误 → 原v4.1 native strict验证、真实新增链、new normal restore和实际selected source/step审计；不恢复旧optimizer/step。
- repeated开发集、单seed、2点best选择和fixedpool分歧 → 配对bootstrap是固定checkpoint条件下未校正逐点CI；groups/warmcold/newlost限制主张，保留Testing开发使用历史。

## Migration Plan

规格strict与root review后实施最小域模块/配置/脚本；聚焦CPU loss/gradient/来源/旧模型回归、完整Hydra resolve和脚本quoting/notes/dryrun/empty/override验证，root再冻结真实运行源码、Mutagen flush三Watching和checkpoint/input身份，通过统一双卡dry-run后按登记句柄训练两臂，不追加探测训练。

两臂完整2000步结束后核验实际2个Validation事件、best Artifact/step/bytes/来源、optimizer两组和scheduler horizon。各selected continued成员与固定source构成一个pool，各做一次完整Evaluation，复用已审计same-policy LIGER wdms8w77及旧v4.3 iy3o3z3q实际输出。独立175raw shards/22363 keys/末商品labels/合法唯一SID/history零重叠/current source/checkpoints/metadata后，复算相对fair LIGER、旧v4.3、treated-control的R/N与pairedCI、new/lost/shared、固定双组/warmcold。

组件保留与整体晋级分开：任一臂对旧iy3o3z3q的R10或N10相对增量≥3%，另一项点值无退化，可保留；CI与groups限制主张。treated-control正向且CI支持才归因history-CE增量，跨0不确认增量；若control独立明显改善可以保留续训贡献而不声称mask有效。

合格部署臂必须同时：R10≥ `1.1*.09694584805258687`（至少2385hits）、N10≥ `1.1*.05404889855718787`、对wdms8w77两paired差值CI下界>0、旧ValidationR10≥.09877029021151008 / N10≥.0511173919307933、actual raw/source审计通过。合格臂中N10较高者唯一入选；精确同N按R、再按treated。都不合格则不Testing、本阶段结束，不改窗口/pair/weight/参数或继续训练，保留有效原历史层和符合标准的正向续训贡献。

若有唯一qualified winner，才安排同policy固定LIGER与选定fixedpool各一次Testing，既有1→全线程最多3。接受要求对新same-policy Testing LIGER双≥1.1、两pairedCI下界>0，并同时通过旧042139al阈值R10≥.07855386128873586（1757hits）/N10≥.04002978116676896；Testing不选择模型、来源或政策。整个新阶段最多2训练/4000steps、2pool完整Validation、2条件Testing，累计训练限额7/34000显式增加而非重置。

## Open Questions

history CE支持对齐是否改善合法目标排序、extra2000续训本身是否有效及怎样改变既有615/401分歧，由这一对匹配臂回答。任何结果都不自动追加预算、扫描或声称训练支持集是原bad case根因。
