## Context

原 residual / mixed / bias / 固定pool均按各自预承诺关闭；5次训练 / 30000 steps已耗尽，Testing已用1次、全线程上限3次。fresh原始训练无重复，Evaluation真实目标均不在本人训练或有效历史中，现有dense路径却未排除个人历史，产生可核验的无效占位。协同关系方向本轮未获支持，新问题只检验固定评分下的历史候选资格；完整五项门禁、真实来源和统计边界见 `docs/copmrec-v4-3-history-exclusion-research.md`。用户在本阶段正式运行前明确：不要求每层立即达到10%，约3%以上明显组件增益可保留累计研究；旧训练额度耗尽不构成用户永久禁止后续研究。

## Goals / Non-Goals

**Goals:** 对固定LIGER和固定pool应用同一有效模型输入历史规则，在全目录原始分数后硬排除历史行，稳定Top10；保留原checkpoint恢复、公共lineage、单卡和标准输出，完成有界same-policy评价。

**Non-Goals:** 新训练、optimizer、graph或train-data构建；评分函数、成员、0.5权重、温度、归一化、history窗口扫描；使用labels、raw training或用户key查找参与政策；只过滤CoPMRec并以未过滤LIGER声称方法增量；恢复旧路线或承诺10%效果。

## Decisions

1. 新文件 `src/recommendation/liger/history_exclusion.py` 承载唯一共享 `apply_history_exclusion(scores, input, catalogmodel) -> masked_scores`，以及 `HistoryExcludedLiger`、`HistoryExcludedFixedLogitPool`。helper输入是完整目录有限FP32分数、`TigerModelInput`和具有原始catalog语义的成员模型；只读取 `input_ids` / `attention_mask`，不读取 `output_keys`。先校验batch / catalog shape、设备、整数SID与二值attention，再把attention=1的完整SID精确映射到当前catalog行，去重后在分数副本中设为 `-inf`。合法行分数逐值不变，不增加参数或评分。
2. 历史边界固定为实际有效输入，最多20个完整商品；正式输入为80 SID tokens / 4 hierarchy。attention必须按完整商品一致，且为右侧padding；半个有效SID、未知有效SID、非二值attention、过长有效历史或不足TopK个可选目录行明确拒绝。attention=0完整padding槽内容忽略，保持原LIGER允许任意inactive token的语义；正式data仍用padding=-1。重复历史商品仅mask一次，cold商品不按全局seen_mask排除，除非它本身出现在该用户有效历史。不得改用完整raw training历史或另选窗口。
3. `HistoryExcludedLiger` 继承原 `Liger`，constructor保留其原kwargs，不引入新旋钮；正式模型dense / Top10 / max_history_items20、candidate及path trace关闭、training_model_config无optimizer/scheduler。继承原 `on_load_checkpoint` 和严格state恢复，不增加persistent checkpoint keys、不改版本标记。仅推理dense retrieve / predict返回原 `(sids, scores)` / 标准ModelOutput；使用原encode / query / dense_logits后调用共享helper，再稳定排序。拒绝非dense、带target_ids的trace、训练loss/eval_step路径、fit / train(True) / training_step / configure_optimizers / checkpoint save；允许原checkpoint正常load。预测开始检查world_size=1。
4. `HistoryExcludedFixedLogitPool` 继承原 `FixedLogitPoolCoPMRec`，constructor原9kwargs及四expected identity不变，原双factory、normal restore、严格state、来源链和参数冻结全复用。只覆盖 `forward`：原父类独立query并等权平均完整 logits后，以 `source_model` catalog调用同一个helper。原 `predict_step` 稳定Top10和训练/多进程/checkpoint拒绝继承，不建立wrapper checkpoint。父 `pool_contract` 原样保留，source/control依旧各自原v4 / v4.1，scale0、multiplier20、cold residual0及原v0链不改。
5. 两个模型property、resolved config和commonwriter metadata统一字段名为 `history_eligibility_contract`，exact9keys固定如下；每次property返回独立dict，不含待学习或待搜索参数。pool metadata另保留原 `pool_contract` 及双Artifact identity，新pool运行wrapper版本为v4.3；LIGER原checkpoint schema不桥接。

```yaml
history_eligibility_contract:
  protocol: copmrec-known-history-eligibility-v1
  history_scope: input_valid_complete_sids_max20
  labels_used: false
  raw_training_used: false
  user_keys_used: false
  score_rule: unchanged_full_catalog_logits_then_history_minusinf
  rank_ties: stable_catalog_row
  cold_items_eligible: true
  single_process: true
```

6. 新薄 model / experiment配置和根 `liger_history_exclusion_inference.sh`、`copmrec_v4_3_inference.sh` 复用统一 `src.main` / Hydra、公共catalog/checkpoint loader、lineage及standard writer。LIGER通过顶层真实 `ckpt_path` normal restore固定best45000；pool顶层 `ckpt_path=null`、双显式checkpoint引用和expectedSHA保持v4.2契约。脚本支持双URI必要参数、quoting、notes、显式dry-run和额外override，单卡NPROC1；formal prelaunch拒绝覆盖冻结评分/历史/来源条件。两臂相同raw split、preprocessing、batch32 / FP32 / seed42、物理GPU2→local0；无新训练入口。新数据组件可薄继承现有LIGER或pool输入配置，保留实际有效历史语义。
7. 唯一生产LIGER checkpoint为 `wandb://baymaxam/GRID/35ig0tz6?role=checkpoint&alias=v1&file=checkpoint_epoch=000_step=045000.ckpt`，SHA `8508e08e2a2cc9ea5d2bbc902aa4e2b6415c8a45b9c8a0ddc879728a6b66b43b`。pool source为 `wandb://baymaxam/GRID/nj9elah1?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt` / SHA `7c52228a665b593b50e45e8aba039f3f4aa4a10ca134d9345c59c59fb142f055`，control为 `wandb://baymaxam/GRID/l3zyr91b?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt` / SHA `4bea4d5771c4f056ad6e0cd69d1204a3220b639868803cbc19aa3332ecf0a3c9`。目录12101商品 / 33cold / SHA `85502adea36aa3d44a61579f1811179bb8d3943980cf190f089dacbaa7fa22e1`，SID / embedding Artifact沿用。生产身份由配置、preflight、实际lineage及独立审计冻结，不在helper中硬编码run或W&B读取。

## Risks / Trade-offs

- LIGER也有历史占位，相同policy会提高其表现 → 新same-policy LIGER是主要分母；旧未排历史阈值仅额外绝对门槛，不能作为单边过滤后的主增量比较。
- 已有Top10只给已知命中位移的条件下界，未知Top11以外可能恢复 → 完整catalog正式预测才计算新效果，既有压缩rank不能当新推荐结果或恢复命中。
- 本Evaluation目标均未见历史不保证所有数据/split一样 → 正式Testing独立读取实际raw标签并报告target是否被政策排除，不把labels用于过滤，不按Testing更改窗口或规则。
- helper可能把部分SID或padding误映射为商品 → 完整SID、attention、未知ID、catalog unique、eligible数量负向CPU测试；合法分数保持与旧模型逐值相同。
- 同一开发集已多次使用、单seed和checkpoint已选择 → 用户paired bootstrap仅是固定checkpoint条件下的未校正逐点区间，保留历史Testing开发使用边界，不声称新原创评分或独立复制。

## Migration Plan

先完成CPU模型/helper、旧模型回归、完整Hydra resolve、脚本参数/语法与OpenSpec strict；root冻结本地实际源码、Mutagen flush成功+三Watching、全runtime字节与真实checkpoint/input身份，按统一路径完成单卡准备。准备不计推荐效果；所有formal运行由root按已登记句柄控制，旧入口可继续运行原政策。

完整Evaluation最多2次：先same-policy LIGER，再同规则固定pool。独立原始175 shards / 22363用户、末商品标签、SID唯一合法、有效历史零输出、catalog / 输入 / checkpoint / source / 实际writer契约审计后，复算R10 / N10、pairedCI、new/lost/shared、固定key组及warm/cold。与各自原输出的政策增量用于说明作用；对新LIGER的公平方法比较决定晋级。

设新same-policy Validation LIGER的R/N为 `R_LV / N_LV`，pool的R/N为 `R_PV / N_PV`。只有 `R_PV >= 1.1*R_LV`、`N_PV >= 1.1*N_LV`、两项pool−newLIGER paired绝对差值CI下界均>0，并且 `R_PV >= .09877029021151008` / `N_PV >= .0511173919307933`（旧Validation门槛2209hits），才能启动至多2次Testing：same-policy LIGER和同fixedpool各一次。门槛计算使用实际新baseline，不预填其指标。

组件保留另行计算v4.3相对自己冻结旧pool i4xwruok的R10/N10点增量：任一相对提升≥3%，且另一项点值≥旧pool，可保留作bad case分析和累计贡献。必须同时记录pairedCI、固定key组、warm/cold与new/lost/shared；CI跨0只支持点值观察，不证明总体增量。该保留判断不替代same-policy公平整体门禁，也不触发Testing或新运行。

Testing累计先前1次→最多3次。新same-policy Testing LIGER记 `R_LT / N_LT`，接受需pool≥1.1倍两指标、两pairedCI下界>0，且同时≥旧042139al阈值 `.07855386128873586`（1757hits）/ `.04002978116676896`。两个Testing仅固定确认，不选择成员、窗口、参数或其他arm。整体门禁未过则停止本轮Testing晋级，独立判断是否保留正向组件；无明显增量或负向按本固定policy边界收缩，不扫参数、不自动扩预算。目标未实际达标保持未完成。

## Open Questions

同一个有效历史政策对LIGER和pool各自的目录外命中恢复，以及policy后pool能否仍超过same-policy LIGER双10%，只能由该有界完整评价回答。未知结果不授权重置5 / 30000旧训练额度或追加任何扫描；未来成本依据新正向证据和bad case另行明确登记，不把旧上限解释为永久研究禁令。
