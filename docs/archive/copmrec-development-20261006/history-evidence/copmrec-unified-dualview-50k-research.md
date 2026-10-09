# CoPMRec v5.1：共享 query 的双目录评分研究记录

## 当前状态

v5.1 完整 Validation 的独立原始输出审计已通过，但双指标至少 +8% 的效果门槛未通过。相对匹配 native42，R10 为 -1.6144%、N10 为 +3.3243%，两项配对绝对差 CI95 均跨零；相对已保留的 v5，两项均下降且 CI95 均为负。停止当前固定 0.5／0.5 双目录机制及权重扫描，保留 v5 的真实正向证据；整体目标仍 active，最后一个训练槽暂未分配，不启动 seed43 或 Testing。

v5.1 唯一正式 seed42 随机初始化连续 50k 训练 [cm0i584p](https://wandb.ai/baymaxam/GRID/runs/cm0i584p) 已 finished／exit0，独立训练审计确认实际 50000 更新、global batch256、完整 100 个 raw Validation 点和已归档的 396 个运行源码文件。训练 job 为 `logs/autonomous/copmrec_unified_dualview50k_train_candidate42_20261005T161444299962Z`，物理 GPU4,7 映射本地 [0,1]。源码 SHA 为 `200a12399a32969f3d5e2d5f7012ff09b272153338dbf6e62795c95b41af77c2`；对应 OpenSpec 为 [add-copmrec-unified-dualview-50k](../openspec/changes/add-copmrec-unified-dualview-50k/proposal.md)。

自己的 raw Val NDCG@10 最优 checkpoint 是 46000 步，训练内 R10 为 `0.08889684081077576`、N10 为 `0.04831480607390404`。已保存的 best 与 `last.ckpt` 都是同一 46000 步状态，165 个 state tensors、154 份 AdamW moments 和 scheduler 均按该步数核验；不存在已保存的 50000 步完整状态文件。实际 50000 更新的预算由完整历史、summary 中的 `trainer/global_step=49999` 和正常 max_steps 终止日志证明，不能将这一预算证明替代终态 optimizer／scheduler 文件验证。证据见 [训练审计](evidence/copmrec-unified-dualview-50k-20261005/training-candidate42.json) 与 [实际 Lightning 保存策略](evidence/copmrec-unified-dualview-50k-20261005/actual-lightning-checkpoint-policy.json)。

独立单卡 Validation [w19eutzj](https://wandb.ai/baymaxam/GRID/runs/w19eutzj) 已 exit0／finished，PID600977，job `logs/autonomous/copmrec_unified_dualview50k_val_candidate42_20261005T181503981072Z`，物理 GPU7 映射 local0、单进程。它使用自己的 best46000，URI 为 `wandb://baymaxam/GRID/cm0i584p?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=046000.ckpt`，SHA256 为 `0779e7ebcfec3f83e8784daed7a49c9f6bbf38a3bf2fe4d65fead03ac5dbe4ff`。独立审计已逐项确认 175 个原始 Evaluation 文件、22363 个用户的 keys／labels、合法且唯一的 Top10、零有效 history 曝光、实际 checkpoint 字节和运行源码，并由同一原始 bundle 复算指标与配对 CI；Testing 未启动。证据见 [完整 Validation 审计](evidence/copmrec-unified-dualview-50k-20261005/inference-val-candidate42.json)、[实际观测](evidence/copmrec-unified-dualview-50k-20261005/observations/val-candidate42-20261005T181654937628Z.json) 和 [Validation job](evidence/copmrec-unified-dualview-50k-20261005/job-val-candidate42.json)。

旧 v5 随机初始化连续训练 50000 更新，自己的 raw Val 最优 checkpoint 为 41000；完整单卡 Validation `8w893ra3` 相对同预算、同 seed、同 history 资格的 native42 `wdms8w77`，R10 +3.5978%、N10 +11.2897%。原双 8% 门槛未通过，NDCG 正向证据保留，不启动 43 或 Testing。实际训练预算由完整 100 个验证点与终态日志独立证明；已保存 best／last 的完整训练状态均到 41000，不能称存在 50000 步完整状态文件。

| 完整 Validation 指标 | native42 | v5 | 相对增量 | 配对绝对差 CI95 |
|---|---:|---:|---:|---|
| Recall@10 | 0.09694584805258687 | 0.10043375217994008 | +3.5977859779% | [-0.0002683003, 0.0073335420] |
| NDCG@10 | 0.05404889855718787 | 0.06015084675195483 | +11.2896809327% | [0.0037752992, 0.0084081903] |

已有排名桶转移显示 Top5 净增 162、6–10 桶净减 84，Top10 净增 78（新增命中 919、丢失 841）。这支持检验头部收益与覆盖损失的取舍，不证明 alpha、残差或某项 loss 是原因，也不支持恢复生成候选补回路线。证据见 [v5 训练审计](evidence/copmrec-unified-50k-20261005/training-candidate42.json)、[完整 Validation 审计](evidence/copmrec-unified-50k-20261005/inference-val-candidate42.json) 与 [v5 研究报告](copmrec-unified-50k-research.md)。

## 完整 Validation 结果与阶段决定

v5.1 完整 Validation 为 R10 `0.09538076286723605`、N10 `0.05584562468008188`。下表的 CI95 是同用户配对的指标绝对差区间，采用 2000 次 bootstrap、seed42，不是相对百分比的区间。native42 是原效果门槛的分母；固定 v5 输出只用于检验本次机制增量，不替换主对照。

| 对照与指标 | 对照值 | v5.1 | 相对增量 | 配对绝对差 CI95 |
|---|---:|---:|---:|---|
| native42 Recall@10 | 0.09694584805258687 | 0.09538076286723605 | -1.6143911439% | [-0.0050529893, 0.0017886688] |
| native42 NDCG@10 | 0.05404889855718787 | 0.05584562468008188 | +3.3242603843% | [-0.0003384832, 0.0039061625] |
| v5 Recall@10 | 0.10043375217994008 | 0.09538076286723605 | -5.0311665183% | [-0.0076476770, -0.0023688682] |
| v5 NDCG@10 | 0.06015084675195483 | 0.05584562468008188 | -7.1573756719% | [-0.0057564397, -0.0027872530] |

相对 v5，新增 Top10 命中 377、丢失 490、净减 113；共享命中上移 503、下移 584。Top5 Recall 下降 6.3620%、NDCG 下降 8.1706%，两项 CI95 也均为负，因此本次固定机制既未增加覆盖，也未保留原头部收益。相对 native42，NDCG@5 仍有 +7.1320% 的正向配对 CI，但不能据此改写 R10／N10 的原双 8% 晋级门槛。上述统计全部来自已保存的同一输出，不产生新推荐列表或额外完整推理。[实际指标、配对比较与排名桶](evidence/copmrec-unified-dualview-50k-20261005/inference-val-candidate42.json)

阶段结论是停止当前共享 query、固定双目录 0.5／0.5 的机制及其权重扫描，保留 v5 的 NDCG／Top5 正向结果和 v5.1 的否证结果。该结果不否定整个 collaborative residual／CF 路线，也不能凭最终排名区分 query 更新、目录几何与梯度变化的各自作用。不追加方法、训练或 Testing；整体目标仍未完成，剩余预算不因本次结题重置。

实际终态与完整结果已完成额外独立核阅：[训练复核](evidence/copmrec-unified-dualview-50k-20261005/training-independent-review.json)、[Validation复核](evidence/copmrec-unified-dualview-50k-20261005/validation-independent-review.json)、[有限bad case](evidence/copmrec-unified-dualview-50k-20261005/bounded-badcase-analysis.json)。[五项路线门禁与累计成本](copmrec-v5-1-stage-decision-20261006.md)保持原双8%目标和最后一个训练槽未分配；后续仅只读检验已有cold错误占位分布，尚未据此分配训练或推理。

## 固定机制与可检验预测

只改变完整目录的评分：记唯一 history query 为 q，内容投影为 p_i，seen 商品残差为 r_i，采用

```text
score(q, i) = normalize(q) · [0.5 normalize(p_i) + 0.5 normalize(p_i + r_i)] / temperature
```

两个目录共享同一次内容投影（包括 dropout）和同一 query，平均向量不再次归一化，因此分数严格等于两路 cosine logits 的固定平均。权重固定 0.5／0.5，新增参数为 0，没有第二个 T5、第二条 history 编码、teacher、续训或 checkpoint 融合。history query 本身沿用 v5 并读取 history residual，因此内容目录分支不是一个独立 native LIGER view。

模型为 [UnifiedDualViewCoPMRec](../src/recommendation/liger/unified_dualview.py)，版本 `v5.1`。其 `unified_scratch_contract` 保留父契约并增加以下 `catalog_scoring`；experiment 顶层与 writer metadata 使用同一字典：

```yaml
protocol: copmrec-shared-query-dual-catalog-v1
views: [normalized_content, normalized_content_plus_seen_residual]
weights: [0.5, 0.5]
renormalize_average: false
shared_query: true
shared_projection: true
```

预测是相对冻结 v5 增加 Top10 净命中，同时保留头部收益。零残差初始化时评分与 v5 相同，cold 商品残差继续为零；训练后的变化需用实际输出检验。若 Recall／净命中不改善，或以明显头部下降换取覆盖，不能称双目录假设获支持。实际部分收益仍按证据保留，不移动原整体门槛，不扫描权重、温度、alpha、seed 或 checkpoint 组合。

## Loss 与梯度的解释边界

原 unit SID CE、unit dense content CE、unit legal-prefix mass mixture NLL 全部保留，alpha 继续由 mixture NLL 学习。固定目录权重 0.5／0.5 与概率混合 alpha 是两个不同量。训练 content CE 仍对 cold logits 使用原 `-100` 规则，mixture 仍使用原合法前缀条件概率；没有加入训练 history CE mask。

dense content CE 与 mixture NLL 都读取同一个双目录最终 logits，并沿 query、projection、残差反向传播。这里没有两套独立 CE head，也没有对内容分支 stop-gradient；mixture NLL 还连接 SID 分支与 gate。SID CE 公式不变，但共享参数的更新改变后，其后续训练轨迹不保证与 v5 相同。

history 输入的原始内容向量在模型中未经 L2 normalization：完整有效 SID 查目录行后，先将内容复制到各 SID token，再调用共享 content projection，加上 seen 残差、SID embedding 和位置 embedding，经过 input LayerNorm／dropout 和 T5 encoder，取最后有效 token 的 hidden state 为 query。query 的 L2 normalization 发生在目录评分时。history 与目录复用 projection 参数，但属于不同调用，训练时不共享 dropout 样本；v5.1 仅在同一次目录调用内复用投影结果，两个目录视图不重复采样 dropout。[history 编码与投影](../src/recommendation/liger/module.py)、[双目录评分](../src/recommendation/liger/unified_dualview.py)

因此“目录代码只改评分”不代表训练后的 query 保持不变：新最终 logits 的梯度仍更新共享 projection、history residual 和 T5 encoder。history 编码结构与 v5 相同，训练后的参数及 query 值却可能不同。只凭最终 SID Top10 的新增、丢失或位次变化，不能拆分 query 与目录的原因。

在零残差处，目录评分对残差的局部偏导为原单目录的一半，history 残差路径保留。这只描述评分函数的局部导数，不代表训练全过程的总梯度或有效 LR 减半。机制同时改变目录几何和训练梯度，即使推荐指标改善，也不能单独归因于“视图互补”或某个 loss。新增参数为零不代表相同 activation、FLOPs 或墙钟成本。

## 训练、选择与预算

薄配置继承 v5，除模型类、版本、任务名和双目录元信息外，训练条件不变：

| 项目 | 固定条件 |
|---|---|
| 初始化／训练 | seed42，推荐模型随机初始化，0→50000 连续 optimizer 更新，无 warm-start |
| optimizer／schedule | AdamW；主干 peak LR 0.0003、残差 0.002；WD 0.035；warmup 2500、cosine horizon 50000、min ratio 0 |
| batch／精度 | 双卡 DDP2，每卡 128、global 256、accumulation 1、FP32、clip 1 |
| 数据／目标 | 原 SID 与 content Artifact；causal max32、有效 history20、sequence80；原三项 loss |
| checkpoint 选择 | 每 500 更新，自己的 raw dense Val NDCG@10 最优 |
| 完整 Validation | 单物理 GPU→local0、单进程；full catalog dense；共同有效 history 排除与 stable catalog row ties |

主对照仍为实际训练 50k 的 native42 `35ig0tz6` 自己的 best45000，以及固定完整 Validation `wdms8w77`；冻结 v5 输出是机制增量对照。原整体目标仍是 R10 与 N10 相对匹配 native 各至少 +8%、两个配对绝对差 CI95 下界为正，并进一步成对复现。首轮绝对点门槛仍为 R10 ≥ 0.10470151589679383、N10 ≥ 0.0583728104417629。当前不具备双 seed 验收结果，不使用 Testing 选择方法。Beauty split 的历史开发使用及 native42 训练源码归档缺口仍保留，不能由新归档回填。

累计上限仍为原阶段 3 次正式训练／150000 更新、3 项完整 Validation、3 次新 Testing，不因版本名重置：v5 与 v5.1 各完成 50000 更新，已消费 2 个正式训练槽／100000 更新；两项完整 Validation 任务均已终态并通过独立原始输出审计；新 Testing 为 0。最后 1 个训练槽／50000 更新尚未分配。本阶段不自动训练 seed43 或启动 Testing，不能把余下 1 槽描述为 native43 与 candidate43 两个模型的预算。后续分配须依据新的实际证据登记。

入口为 [训练脚本](../copmrec_unified_dualview_50k_train.sh) 和 [推理脚本](../copmrec_unified_dualview_50k_inference.sh)，均通过统一 `src.main`；默认训练双卡、推理单卡，实际物理卡位由 root 按资源登记，不停止其他任务。

## 已完成与待完成验证

architecture 报告新核心 27 项 CPU 测试及原 v5 45 项回归通过，Ruff／格式检查通过；配置 owner 实跑新配置／脚本 40 项聚焦测试通过，覆盖完整 Hydra resolve、父训练参数相等、Bash 语法、notes 两种形式、quoting、空值／错误参数、额外 override、dry-run、推理单卡 guard 和 checkpoint 原 URI 元信息。独立配置／脚本 review 通过，OpenSpec strict 通过。这些是实现证据，不是正式效果或实际 smoke 证据。

新增运行文件为模型 1、组件／experiment 配置 3、根脚本 2，共 6 个；实际清单 396 文件，source SHA `200a12399a32969f3d5e2d5f7012ff09b272153338dbf6e62795c95b41af77c2`，旧 390 逐文件字节一致。官方 Mutagen flush/status 成功，三个 session 均 Watching 且无 conflict。真实 CPU 预检确认 params11031809／165 state tensors／AdamW 两组空状态，实际两个上游输入和评分契约已核。GPU6 后来忙碌使首次 smoke 启动在 job 创建前被 guard 拒绝，0 run；保留原6,7准备与CPU记录，重新准备4,7并做实际CPU预检。唯一实际 smoke PID171722 正常 exit0，两个 rank 及恰好一步通过，0 W&B run。两个只读审计器初始分别通过54／80项纯检查；正式训练审计与完整单卡 Validation 原始输出审计均已完成，后者验证执行与产物有效，但效果门槛未通过。凭据见 [implementation verification](evidence/copmrec-unified-dualview-50k-20261005/implementation-verification.json)、[smoke verification](evidence/copmrec-unified-dualview-50k-20261005/smoke-verification.json)、[formal job](evidence/copmrec-unified-dualview-50k-20261005/job-train-candidate42.json) 和 [Validation 审计](evidence/copmrec-unified-dualview-50k-20261005/inference-val-candidate42.json)。

训练审计器首次实际调用在本地 build_payload 阶段因 `runtime_sha256` 与实施凭据真实字段 `runtime_source_sha256` 不符而退出，发生在 SSH 前。原审计器字节已保留；随后修正本地 schema 读取并核真实 payload 和 4 项负向夹具，使用纠正版完成训练审计。原实施凭据保留原审计器指纹 `35ad0899…0737e`，纠正版为 `8ac54079…f0206`，不能声称原审计器字节未变；该修正没有改变运行源码、配置、训练 job、checkpoint 或预算。完整变更边界见 [implementation-schema-fix-verification](evidence/copmrec-unified-dualview-50k-20261005/implementation-schema-fix-verification.json)。
