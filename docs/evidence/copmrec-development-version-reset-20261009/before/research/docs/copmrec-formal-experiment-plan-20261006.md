# CoPMRec v5.3 正式实验计划

2026-10-08：阶段C以[BMX-117 M3统一协议](https://linear.app/baymax104/document/copmrec-v53m3-消融与机制实证协议2026-10-08-643847ec8183)为准，Beauty/seed42五变体、5训练/250k/0额外独立Validation/5Testing已完整核验；三个机制证据包全部交付，4/4 diagnosis任务、含失败工程attempt总5，M1固定bundle分析另计1次且零model-forward。精确run/URI/SHA/指标/CI见[机器状态](../research-state.yaml)及[最终实证汇总](../../GRID/docs/evidence/copmrec-m3-completion-20261008/m3-empirical-summary.json)。主矩阵9/9和原LIGER复用范围保持；M3只提供实证观测，不按方向评判或扩预算。

2026-10-08 BMX-116正式主矩阵已核验完成：9/9新CoPMRec训练、9/9Testing，9/9原LIGER hybrid配对输出独立核对，原baseline保持复用。已完成checkpoint lineage、testing用户/标签、合法唯一Top10、CoPMRec历史排除与四指标独立复算；九个配对bootstrap及三seed均值/标准差已归档。原LIGER历史政策与CoPMRec不同，保留完整方法比较边界。详见GRID docs/evidence/copmrec-main-completion-20261008/；dense内部对照与消融不计本父任务完成。

| Dataset | CoPMRec NDCG@10 mean +/- std | LIGER hybrid NDCG@10 | Gain | CoPMRec Recall@10 mean +/- std | LIGER hybrid Recall@10 | Gain |
| -- | -- | -- | -- | -- | -- | -- |
| beauty | 0.04355895 +/- 0.00109371 | 0.02992889 | 45.54% | 0.07583956 +/- 0.00269898 | 0.05570213 | 36.15% |
| sports | 0.02398604 +/- 0.00054744 | 0.01574162 | 52.37% | 0.04356986 +/- 0.00067478 | 0.02864393 | 52.11% |
| toys | 0.04668824 +/- 0.00031526 | 0.02629864 | 77.53% | 0.07699705 +/- 0.00066238 | 0.04837214 | 59.18% |


以下运行快照保留当时事实，当前完成状态以本文顶部9/9核验、机器状态和Linear为准。历史训练审计：seed42/200共6个训练finished并通过固定50k及best核对，running 0；当时Testing与seed2026尚未启动。六个独立Testing命令已填入各子issue的精确best v0 URI及SHA256并验证脚本参数/Hydra配置。详情见GRID docs/evidence/copmrec-training-audit-20261007/。

2026-10-07 最新运行状态：seed42三个训练均finished、退出码0，best审计与Testing待完成；本轮按用户明确授权启动seed200三个双卡tmux训练，并允许共享显存足够的GPU。累计训练started 6 / finished 3 / running 3，Testing启动与完成仍0；seed2026待后续启动。启动回执：GRID docs/evidence/copmrec-tmux-seed200-20261007/。

## 本轮目的与身份

用户于 2026-10-06 确定 v5.3 为正式 CoPMRec，并要求回退 Milestone 2、按新计划重新执行；2026-10-07 明确正式 baseline 为 **LIGER hybrid**，**LIGER dense 仅作内部对照**，不以 v5.2 作为论文主比较。

本轮确认冻结方案的整体效果，并以有限对照测量共享残差和训练监督的作用。不重新寻找方法、扫描 alpha、修改排序器或扩大 baseline 名单。CoPMRec 开发阶段 run 只进入历史文档；新正式主矩阵现已9/9完成核验，新M3观测按独立预算登记。2026-10-07 按用户纠正恢复原 LIGER hybrid 的9组已完成正式结果：BMX-133～141及父任务BMX-132恢复原始标题、描述、Done/有效，不再要求重新训练或独立Validation。

本文件本身不授权agent自动启动实验。2026-10-07用户另行授权在node1以tmux启动Beauty/Sports/Toys seed42三个双卡训练，并允许共享显存充足的GPU；已启动3、完成0，Testing未启动。运行身份为BMX-120/gshpyn49、BMX-119/jk4zk19n、BMX-17/hs72xkan；其余六单元不在本次启动范围。详见GRID `docs/evidence/copmrec-tmux-train-launch-20261007/`。配方见[版本定义](copmrec-formal-version-20261006.md)，预算按实际训练与完整评价计数。

## 范围与研究问题

沿用已确定范围：Beauty / Sports / Toys × 训练 seed **42 / 200 / 2026**。论文五方法仍为 CoPMRec、LIGER、TIGER、LETTER、SASRec。COBRA 等不进入本轮；不添加 backbone、数据集或用于挽救结果的第 4 个 seed。

| 问题 | 必要比较 | 解释边界 |
|---|---|---|
| RQ1：完整方法是否优于核心基线？ | CoPMRec v5.3 对匹配 LIGER hybrid，9 个单元 | 整体比较不以单组件相对内部版本的显著性为前提 |
| RQ2：在既定方法范围内处于什么位置？ | 五方法主表，共 45 个评价槽位 | 保留外部方法原生关键机制；不混用论文原始数字 |
| RQ3：训练设计对应怎样的 dense 观测差异？ | Beauty/seed42的A1～A5五个固定训练对照 | 只报告观测值与带符号差，不将单seed组件观测外推为跨seed稳定性 |
| RQ4：命中、排名和内部概率如何变化？ | M1可加分解、M2固定checkpoint残差2×2、M3真目标前缀概率 | 重训与评分干预分别解释；条件化概率不代表free-running候选恢复 |

正式主表比较既有 LIGER hybrid 的原生检索流程与新 CoPMRec v5.3 的联合 dense 部署。LIGER原训练配置为双卡50k/global256/FP32，原best与Testing继续复用；进入配对分析时核对数据、上游输入、骨干与选点。保留各自推理机制，不声称相同检索支持集、候选数或推理计算量。LIGER dense 使用既有基线的同一best checkpoint作内部诊断，不替代正式baseline。

原问题中的前缀内容概率机制保留在训练目标和研究动机中。v5.3 部署是 dense；旧 mass/max/legal 的 beam 机制表不能作为本轮 dense 效果完成证明。若为正文保留路径诊断，须作为另行准备的受控诊断，不改变主部署、不冒充主结果。

## 阶段 A：冻结输入、基线协议和可执行入口

此阶段不产生推荐效果，不标记任何训练单元 Done。

1. 对每个数据集登记 train/evaluation/testing 数据 manifest、规模与文件 SHA-256；明确 evaluation 即 Validation、testing 即 Testing。
2. 冻结 SID 和内容向量 Artifact ID/digest 或本地 manifest/分片哈希；三个训练 seed 使用同一数据集输入。既有上游产物可技术复用，但必须审计且不计入本轮推荐完成数。
3. 核对 SID 冲突、item ID 与目录行映射、训练 seen_mask、cold 定义、标签分离及有效历史截断。Validation/Testing 同一用户必须使用一致 keys 和合法商品支持集。
4. M4 LIGER hybrid 的9个单元复用已有正式训练、Validation-selected best和Testing，恢复原Done/有效。原SID CE/content CE、original生成20与全部cold并集、内容终排及candidate trace保持原协议。2026-10-07只读核验18个run均finished、best Artifact可访问、hybrid配置和指标与原issue一致；不新增训练或完整评价。回执：GRID `docs/evidence/liger-baseline-restore-20261007/`。
5. CoPMRec仍使用有效输入最近20件历史排除；既有LIGER Testing使用原Liger实现，不能改写成已运行过`HistoryExcludedHybridLiger`。接入新主表时核对历史资格、split与用户keys；如需共同资格重评分，复用原best，单独登记必要评价成本，不据此重置原issue或要求重训。本次不自动增加或启动重评分。
6. TIGER、LETTER、SASRec 已有效正式结果不因 CoPMRec 版本更新自动回退。进入主表前仍审核数据、split、资格规则、上游和选点；需要重新评分时计入真实完整评价次数，不默认为零成本。
7. 正式入口使用 GRID 根目录脚本、`uv run` 与统一 `src.main`。训练两卡，推理单卡；运行前完成 Hydra compose、脚本 dry-run、Mutagen flush 与 runtime source 快照。

### 各数据集输入登记表

下列 URI 为既定上游技术依赖，不更换开发阶段已冻结的输入。生产来源已登记，但本轮运行前仍须核验实际 Artifact digest、数据身份和代码消费记录；其保留不计为任何推荐训练/推理完成。缺失哈希保持 `null`，不能填为“本轮已审计”。

| 数据集 | SID 引用/digest | 内容引用/digest | 数据 manifest/hash | 上游状态 | Testing 使用史 |
|---|---|---|---|---|---|
| Beauty | `wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt`；digest待核 | `wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt`；digest待核 | `null` | 量化器 `1ysekezs`；本轮消费前审计 | 已有开发使用；按历史登记 |
| Sports | `wandb://baymaxam/GRID/jcyr5l3p?role=semantic_id&file=merged_predictions_tensor.pt`；digest待核 | `wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&file=merged_predictions_tensor.pt`；digest待核 | `null` | 量化器 `pjrhwr04`；本轮消费前审计 | 待审计；不预设未消费 |
| Toys | `wandb://baymaxam/GRID/3dycz43g?role=semantic_id&file=merged_predictions_tensor.pt`；digest待核 | `wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&file=merged_predictions_tensor.pt`；digest待核 | `null` | 量化器 `dt03m6xm`；本轮消费前审计 | 待审计；不预设未消费 |

上游引用、实际运行命令及 best checkpoint 未核验时保持缺失，不能编造示例 Artifact 成为真实命令。

### 正式 CoPMRec 人工运行入口

从 node1 的 GRID 仓库根目录执行。以下统一命令显式使用上述固定输入，`DATASET` 和 `SEED` 对应单元表；训练示例为物理 GPU 2、3 → local `[0,1]`，推理示例为物理 GPU 2 → local `[0]`。用户可按空闲设备改物理编号，模型配置不随设备改变。正式脚本仍支持 `--dry-run` 和 notes；dry-run 不计为正式实验。

```bash
DATASET=beauty
SEED=42
case "$DATASET" in
  beauty)
    SID="wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt"
    CONTENT="wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt"
    ;;
  sports)
    SID="wandb://baymaxam/GRID/jcyr5l3p?role=semantic_id&file=merged_predictions_tensor.pt"
    CONTENT="wandb://baymaxam/GRID/psec3u5i?role=semantic_embedding&file=merged_predictions_tensor.pt"
    ;;
  toys)
    SID="wandb://baymaxam/GRID/3dycz43g?role=semantic_id&file=merged_predictions_tensor.pt"
    CONTENT="wandb://baymaxam/GRID/d1q00dco?role=semantic_embedding&file=merged_predictions_tensor.pt"
    ;;
  *) echo "unsupported dataset" >&2; exit 2 ;;
esac
CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 bash ./copmrec_train.sh \
  --data-dir "data/$DATASET" --dataset "$DATASET" --seed "$SEED" --devices '[0,1]' \
  --semantic-id-path "$SID" --embedding-path "$CONTENT" \
  --group "paper_main_copmrec_$DATASET" \
  --notes "formal CoPMRec v5.3; fresh scratch50k; ${DATASET}/seed${SEED}"
```

训练完成且 own-best 已审计后，直接执行单卡完整Testing，不额外运行独立Validation。训练期间每500步的validation与best选点保留。下方命令需填入该单元真实checkpoint URI与SHA-256；占位符不能直接执行，也不能填开发checkpoint。示例为Beauty/42，其余单元同样显式绑定固定输入。

```bash
COPMREC_BEST_CKPT="<本轮Beauty/42正式训练的Validation-selected checkpoint URI>"
COPMREC_BEST_SHA256="<该checkpoint实际SHA-256>"
SID="wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt"
CONTENT="wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt"
CUDA_VISIBLE_DEVICES=2 NPROC_PER_NODE=1 bash ./copmrec_inference.sh \
  --data-dir data/beauty --dataset beauty --seed 42 --devices '[0]' --split testing \
  --semantic-id-path "$SID" --embedding-path "$CONTENT" \
  --group paper_main_copmrec_beauty \
  --checkpoint "$COPMREC_BEST_CKPT" checkpoint_sha256="\"$COPMREC_BEST_SHA256\"" \
  --notes "formal CoPMRec v5.3 Beauty/42; frozen Testing; no selection or tuning"
```

LIGER hybrid 使用恢复后的原issue内容：训练和Testing命令仅供复现，已有run、best和指标直接登记，不要求重新运行。原始Testing不套用新增的历史资格wrapper。共同资格适配代码的准备状态见[代码验证回执](../../GRID/docs/evidence/copmrec-formal-release-20261006/local-code-verification.json)，准备通过不等于旧结果使用了该实现。CoPMRec新checkpoint仍待新训练与own-best审计。

## 阶段 B：Milestone 2 的 9 个新正式 CoPMRec 单元

每单元完整链路为：**从头双卡50k训练（含validation选点）→ 审计自身best → 单卡Testing → 独立复算与配对分析**。2026-10-07按用户要求统一实验issue模板，不额外运行独立Validation；每单元只有1训练run与1Testing run。

| 单元 ID | 数据集 | seed | Linear issue | 新训练 | 额外 Val | Testing | 当前状态 |
|---|---|---:|---|---:|---:|---:|---|
| C-B42 | Beauty | 42 | BMX-120 | 1 | 0 | 1 | training_running：gshpyn49 |
| C-B200 | Beauty | 200 | BMX-121 | 1 | 0 | 1 | pending |
| C-B2026 | Beauty | 2026 | BMX-118 | 1 | 0 | 1 | pending |
| C-S42 | Sports | 42 | BMX-119 | 1 | 0 | 1 | training_running：jk4zk19n |
| C-S200 | Sports | 200 | BMX-15 | 1 | 0 | 1 | pending |
| C-S2026 | Sports | 2026 | BMX-16 | 1 | 0 | 1 | pending |
| C-T42 | Toys | 42 | BMX-17 | 1 | 0 | 1 | training_running：hs72xkan |
| C-T200 | Toys | 200 | BMX-18 | 1 | 0 | 1 | pending |
| C-T2026 | Toys | 2026 | BMX-19 | 1 | 0 | 1 | pending |
| 合计 | 3 数据集 | 3 seed | 父任务 BMX-116 | **9** | **0** | **9** | **完成 0/9** |

执行顺序固定为各数据集 seed42 的配对单元先完成数据/产物有效性审核，再执行其余 200/2026。CoPMRec 在任一数据集不得因 Validation 结果换回 v5.2、改辅助权重或新增训练日程。

### 固定训练与选点

- 推荐器全部从头初始化，`ckpt_path=null`；不加载开发推荐权重。
- 50k 更新，global256，两卡每卡128，FP32；主 lr 0.0003、残差 lr 0.002、wd0.035；warmup2500/cosine50000。
- 四项损失权重均为1；learned global alpha从0.5开始；目录内容/SID固定；历史与目录共享残差，cold残差0。
- raw full-catalog dense `val/ndcg@10` 选点，每500步验证；并列取首次最佳。禁止按独立 Testing 改选 checkpoint。
- 保存源代码真实运行字节、训练完整配置、实际更新证据、曲线、best Artifact及 checkpoint SHA-256。未保存终态 full-state 时如实说明，不把 best 的 global_step 写成完成更新数。
- 显式记录物理 GPU 到 CUDA local index 的映射，实际资源选择不影响冻结模型超参数。

### 固定评价与分析

- 单卡单进程、全目录评价，不采用sampled negatives。CoPMRec给全目录联合评分并排除有效输入最近20件历史商品；LIGER保留既有original hybrid与内容终排。历史资格差异在配对协议审计中明确记录，不能未经核对就声称两臂完全一致。
- CoPMRec历史排除后按稳定目录行tie-break；LIGER原排序及cold资格按实际run记录。合法、唯一Top10与固定bundle `keys/predictions`协议由实际产物核验；不改写旧预测来迎合新实现。
- 报告每seed Recall/NDCG@5/@10、3seed均值±样本标准差及对应 LIGER hybrid 差值，禁止只取最佳seed。
- 同seed/数据集用相同用户keys做配对bootstrap：NumPy PCG64 seed42、2000次、95%未校正区间。用户CI只反映固定checkpoint的用户采样不确定性；不能替代训练seed稳定性。
- 跨seed描述须保留3个训练结果；若计算按用户的平均seed差CI，先按key平均，再重采样用户，不把用户×seed当独立样本。
- 主指标NDCG@10，Recall@10为方向保护。报告效应量和区间，不把历史双8/双10探索门禁变成本轮修改模型的阈值。
- 正向结果支持所测设置下整体效果；负向结果降低相应优越性主张；不确定结果保留区间与限制。所有有效单元进入报告，不因结果不利新增扫描、换seed或排除单元。
- 有效性失败单独记录失败成本和原因，不作方法反证；用户手动决定重跑，不抹掉失败attempt。

## 阶段 C：有限训练消融与新的机制分析

主方法保持v5.3，阶段C唯一有效协议为 `copmrec-m3-v53-20261008-v1`，详见 [Linear M3统一协议](https://linear.app/baymax104/document/copmrec-v53m3-消融与机制实证协议2026-10-08-643847ec8183)。旧三臂×三数据集清单已替代，历史开发Legal/Max/Mass不作为本阶段来源。整体多数据集、多seed验证由完成的正式主矩阵承担；本阶段只固定Beauty/seed42，结果不影响数据集或seed选择。

记Full为S+J+M+N，S/J/M/N分别为SID CE、joint catalog CE、mixture NLL、native-view CE，G为真目标前缀下合法生成NLL。

| 实验 | Issue | 精确定义 | 新执行量 |
| -- | -- | -- | -- |
| A1 NoMixture | BMX-122 | S+J+N，alpha冻结0.5、不进优化器 | 1 train / 50k / 1 Testing |
| A2 NoResidual | BMX-123 | 历史/目录残差恒0，S+J+M+N保留；J=N重复项保留 | 1 train / 50k / 1 Testing |
| A3 NoNative | BMX-129 | S+J+M | 1 train / 50k / 1 Testing |
| A4 LegalGenReplace | BMX-142 | S+J+G+N，G按用户×4层平均，alpha冻结 | 1 train / 50k / 1 Testing |
| A5 JointCEReplace | BMX-143 | S+2J+M，第二个J复用同一次logits/投影/dropout | 1 train / 50k / 1 Testing |
| M1 命中/rank可加分解 | BMX-144 | Full+5变体的6份Testing bundle，固定seen/cold、training频次、历史长度切片 | 已交付；0 model-forward，114 slice / 608 CI |
| M2 残差2×2评分干预 | BMX-145 | 固定Full own-best，h=0重算query，目录c=0对应native评分；核对V11复现 | 1 diagnosis / 3新增评分pass |
| M3 真目标前缀概率 | BMX-146 | Full/A1/A4 own-best的逐层概率/NLL/rank/entropy/JS；只有Full报告mix/alpha | 3 diagnosis / 3前缀pass |

五个变体均scratch，沿用50k/DDP2/global256/FP32；每500步训练内raw dense Validation，选各自首次最高val/ndcg@10，然后一次完整单卡Testing，不增加独立Validation。Full复用Beauty/seed42的gshpyn49/vosmuihm，不新训练或Testing，禁止作为变体初始化。累计固定 **5训练 / 250k updates / 0额外独立Validation / 5Testing**；另 **4 diagnosis任务 / 6全量forward等价pass**。M2 V11复现属于额外核验开销，单列实际成本。

专用变体、checkpoint/optimizer契约及三类统一Trainer.test诊断已验证；训练/Testing group为`paper_ablation_copmrec_beauty`，诊断为`paper_mechanism_copmrec_beauty`。八issue均使用统一模板和独立可复制命令；2026-10-08用户授权启动并在五训练完成后授权继续。5/5训练完成50k更新和100个训练内选点，各自own-best精确URI/SHA/最早最大值/optimizer与source已核验；5/5单卡Testing全部exit0，N=22363，预测合法唯一且排除输入历史，四指标与W&B一致。实际train/test ID、选点、硬件、指标和CI见[当前计划](../ideas/current-plan.md)与[最终实证汇总](../../GRID/docs/evidence/copmrec-m3-completion-20261008/m3-empirical-summary.json)。没有新增训练或独立Validation。

M1 `m3hitc8p`独立核验六臂134178 user-variant、114 slice、608 CI及132确定性案例，14项analysis Artifact文件逐path/size/base64MD5/SHA一致，0 model-forward。M2 `717vgnkn`完成四bundle、76 slice/304 CI、V11精确复现Full及cold同query目录开关logit不变，12项文件身份一致。M3 Full `j2rworuj`、A1 `5murlou7`、A4 `pltnvpjf`均完成每源89452前缀记录、各8项文件身份核验，3/3来源已交付。实际4/4 diagnosis任务，工程attempt总5包含初次`ls14h0dw`序列化失败；M1另计一次bundle分析。失败attempt保留不计完成，metadata转换修复的19测试/Ruff/strict/flush与retry开销独立登记，不扩大训练预算。

五训练runtime source仍为`42f0d7c7dce0a09961e0c8c23f9ba35982abfb796d13952c3f17eaeba17c9f2e`；Testing/diagnosis为`5e95cf7e87b28b730976b5a6d279acdc7aca3a270a08dd5e4519ce1fc6624867`、origin verified，仅两处metadata序列化文件变化，不追改训练源码。原始四指标、五Full配对及A4−A1/A5−A3、K5/10三项可加贡献、前缀三源均值与分位数绑定精确URI/Artifact version/digest/MD5/SHA。配对bootstrap为PCG64/seed42/2000/95% pointwise，不作多重比较校正；单训练seed用户重采样不能表达跨seed稳定性。资源只记录已有W&B runtime，双卡设备小时为分配时长估算，active GPU时间与peak VRAM未测得，不补专用效率实验。

只交付观测值、带符号差、样本数、pointwise CI、工程核验及解释边界，不给好坏或积极/消极标签，不据结果晋级、否决或自动扩预算。Done表示预定证据已完整交付，与方向或显著性无关。native query仍有历史残差；M2固定checkpoint干预不等同A2重训；M3 teacher forcing不等同自由beam路径恢复，正式部署仍为joint dense。

### LIGER dense 内部对照的独立计数

固定9个数据集/seed条件，全部复用既有对应LIGER hybrid的own-best，不增加训练。它属于内部分析，不改变已恢复的hybrid完成状态。

固定仅做Testing，单列新增9次完整评价；不增加训练和Validation。内部输出标记`evidence_phase=internal`、tag `copmrec-internal-v53`，从既有对应正式LIGER own-best运行。当前未启动，独立于CoPMRec的9次Testing计数；命令另行准备，不写入已恢复的原LIGER issue冒充原内容。

## 阶段 D：五方法主表与论文材料

共45评价单元：五方法×三数据集×三seed。这是**表格槽位**，不是45次新增训练。CoPMRec的9单元从头新训练（含validation选点）及新Testing，不额外Validation；LIGER原9组正式结果复用。所有既有baseline接入新主表前核实际协议，原完成与新CoPMRec完成分开计数。

外部方法保留原方法的关键架构与训练目标；共同数据、全目录评价、split和历史资格须核对。其训练预算差异、SID依赖、调参来源和计算成本明确标注，不强行声称五方法全部等FLOPs。仅CoPMRec/LIGER主配对要求冻结的50k/global256协议。

图表与正文只从本轮有效正式结果或审计通过的既有正式baseline生成；开发主方法数字不得进入正式主表。候选生成、冷启动、效率、辅助CE独立增量的主张各自受证据约束；不因论文需要而补造优势。训练runtime/GPU数作为复现信息记录，不增加专用效率实验。

## 本轮计数与完成门禁

| 工作包 | 确定新训练 | 确定额外Val | 确定新Testing | 已完成 |
|---|---:|---:|---:|---:|
| M2 CoPMRec正式9单元 | 9 | 0 | 9 | 9；2026-10-08独立核验完成 |
| M4 LIGER hybrid正式9单元 | 0；复用原训练 | 0；原训练内已选点 | 0；复用原Testing | 9；原Done/有效恢复 |
| 同既有LIGER checkpoint的dense内部对照 | 0 | 0 | 9；内部对照单列 | 0；不作正式baseline完成门槛 |
| M3五变体×Beauty/seed42 | 5 | 0 | 5 | 5/5训练50k与own-best、5/5 Testing及独立指标复算完成 |
| M3机制M1/M2/M3 | 0 | 0 | 0；另4 diagnosis / 6新增评分或前缀pass | 三包完成；4/4 diagnosis、工程attempt5含1失败；M1零forward分析另记1 |
| TIGER/LETTER/SASRec缺口 | 审核后明确 | 审核后明确 | 审核后明确 | 不自动重置已有效正式结果 |
| 预测复用的分析/写作 | 0 | 0 | 0 | BMX-144六臂分析已核验；论文材料按正式证据生成 |

主配对成本为CoPMRec **9×50k=450k更新 / 0额外Validation / 9Testing**，已完成；训练内validation与选点包含在train run中。LIGER原9组训练/Testing复用。dense内部另列9Testing，不计本次BMX-117授权。M3另列 **5训练/250k更新/0额外Validation/5Testing** 及 **4 diagnosis/6新增forward等价pass**。必要共同资格重评分另列成本，不自动重训或扩预算。

一个M2 issue在自身新正式训练、训练内validation-selected best审计、Testing、输出合法性与独立指标复算闭合后Done，不要求独立Validation run。父BMX-116在9个子单元全部验收后Done。旧完成记录保留历史。

登记包含单元ID、method/version、dataset、seed、status、training_run、testing_run、训练内validation选点证据、checkpoint引用/哈希、输入身份、runtime source、实际更新、硬件、metrics与审计链接。validation_run保留null并标记not_required，不能当作待完成任务；其他pending项使用null。

## 当前执行状态

- 正式版本、矩阵与证据边界冻结；BMX-116九主单元与BMX-117八issue证据均完整核验。M3五训练/五Testing完成，M1零forward、M2固定残差干预、M3三前缀来源全部交付；真实命令与资源未超出原预算。
- 主实验group与其他方法统一为`paper_main_<method>_<dataset>`：CoPMRec为`paper_main_copmrec_beauty/sports/toys`，同数据集各seed及训练/Testing共用group；版本、seed、split与正式身份由config/notes/tags区分。脚本默认值、issue和本文命令一致。
- CoPMRec主矩阵训练9/9、Testing9/9完成；LIGER既有9/9完成保持。消融实现与命令验证已完成，新运行与实证结果独立登记，未完成项不因授权或准备完成而记Done。
- CoPMRec使用本轮自身新best；LIGER使用原正式训练的已选best。上游、split/keys与评价资格接入主表前按真实产物核对。
- LIGER共同历史资格适配仅为已有代码能力，不描述为旧Testing的实际协议；必要重评分另行准备。dense内部固定9完整Testing单列，不替代hybrid主baseline。
- CoPMRec issue统一为实验单元、当前状态、固定协议与输入、训练/Testing命令、正式结果、Done门禁；每单元仅两条命令，额外Validation要求已移除。LIGER原issue与完成状态保留。模板更新回执见GRID `docs/evidence/copmrec-issue-template-20261007/`，原LIGER恢复回执见 `docs/evidence/liger-baseline-restore-20261007/`。
