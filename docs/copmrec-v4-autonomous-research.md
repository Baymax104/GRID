# CoPMRec v4自主改进阶段

## 当前状态：预算耗尽，保留残差正向结果，目标未达

本阶段已关闭，状态 `closed_budget_exhausted_positive_residual_below_goal`：3次训练共18000步全部完成，固定mixed规则负向关闭，不再追加残差/LR/mixed实验。已验证deploy为 `z9envq1n` best5000的单卡dense Testing `1edkvgk5`，相对固定真实LIGER dense的R10/N10为+5.2599%/+5.6354%。当前开发起点 `nj9elah1` best6000的完整Validation `mq8hhof5` 为+6.9223%/+7.1520%，其Testing未运行；相对旧v4的两项配对CI均跨0，未确证额外LR收益。

不消费第二Testing槽。全线程两指标相对+10%目标保持active、尚未完成；新问题仅只读证据与路线门禁，未授权新阶段启动。完整数值、来源和关闭决定见后文；以下按时间保留阶段开始时的假设、预算与实际进展。

## 阶段开始时的用户授权与目标

2026-10-05用户明确要求长时自主改进，以CoPMRec v0为基础，可增加模块或调整方法，允许SSH启动node1训练/推理。优先已有checkpoint续训、多卡训练，推理必须单卡。此授权覆盖本阶段启动行为，旧阶段manual-only/零预算记录保持历史事实。本线程目标持续为相对真实LIGER dense推荐指标至少10%增益，不把实现或运行启动视为完成。

验收采用Beauty/seed42、原Testing22363用户、完整SID精确匹配，Recall@10与NDCG@10相对固定基线均≥10%。基线run042139al/train35ig0tz6/best45000的dense口径：Recall10=0.071412601172、NDCG10=0.036390710152、1597命中。阈值Recall10≥0.078553861289（至少1757命中），NDCG10≥0.040029781167。配对用户区间、原始用户/标签、输出合法性、checkpoint/source/input lineage都需核验。既有Testing参与开发的边界保持，后续参数及checkpoint选择只用evaluation。

## 阶段开始时的证据与路线判断

当前保留的v0 6dspa7e3/best48000有1608命中，R10=0.071904485087、N10=0.036448314264，相对LIGER dense仅+0.689%/+0.158%。v0自身dense遗漏97、超dense增量90，净差−7；假设无损补回全部97仍仅1705命中，不足1757。已有候选覆盖但最终未命中853，其中warm719；warm第11–15有406例。这提供商品判别空间，但oracle不是可部署收益。

v3根Max恢复首层遗漏，却伴随后层损失与dense退化；v3.1补正净增11，未有稳定收益；固定推理alpha0.813对v3.1召回不变；固定0.8从头训练净−49，其中dense−46。以上否定“只减少dense准入遗漏就能取得所需净收益”的当前实现晋级，不否定生成式候选全部潜力；不恢复归档动态/decoder路线结论。

本阶段转向的可反驳假设：共享内容映射缺少完整商品身份的协同自由度，zero-init训练商品残差能在保留v0初始行为后增强商品区分，并通过共享logits改善最终hybrid。若匹配续训没有验证改善，停止这个具体残差设置；不以新命名或反复Testing扫参維持主张。

## 方法与相关工作定位

`z_i=content_projection(content_i)+seen_i*r_i`，新增r_i全零、不消耗随机数，cold乘零。共享到历史token输入、目录cosine分数、content CE、mixture NLL和最终排序。SID/内容/混合loss仍为各1、alpha可学习、Mass beam20+cold。新增约12101×128=1.55M参数，费用须明确报告。

当前v0和LIGER已经有全目录content CE，不将普通CE/InfoNCE作为新目标。LIGER官方实现有`ground_truth+item_id`的ID+text表示，hybrid限制不同，本方案是GRID v0的扩展，不能宣称首创ID残差或仍保持纯SID参数规模。[LIGER论文](https://arxiv.org/html/2411.18814v2)、[官方评价源码](https://raw.githubusercontent.com/facebookresearch/liger/main/src/evaluation.py)。商品级生成竞争监督有[LOHRec](https://aclanthology.org/2025.findings-emnlp.977.pdf)等先例，暂不实施该独立模块。

## 最小阶段实验与累计预算

首批两个串行双卡训练臂，各从v0 best48000 weights-only初始化，optimizer/scheduler/global_step重置；每臂6000步，全球batch256，FP32，主干LR1e-4、残差LR1e-3、weight_decay0.035、warmup300、cosine6000、梯度裁剪1、seed42。按最终hybrid evaluation NDCG10选best，记录Recall10。物理GPU2,4→logical0,1；正式推理物理GPU2→logical0。

1. 试验臂：residual_scale1，检验新增协同自由度是否能改进部署指标。
2. 匹配对照：residual_scale0冻结零残差，其余配置和训练曝光一致，区分额外6000步的收益。
3. 预留一次训练槽，仅在已有验证正面依据下用于独立确认或有决策价值的针对性改动；不是预先授权无差别扫参。单次上限6000步。

阶段累计训练最多3次/18000步；每个验证选择结果至多一次单卡Testing，最多3次，不重置旧budget或逐轮刷新。真实硬件失败的重试单独记录实际成本，不隐匿失败run。正向：验证集超过门槛后做Testing与独立复算；负向：停止当前具体设置；不确定：使用剩余确认槽需有positive validation依据。任何结果不能降低全线程10%目标，未达标保持active，并只按新证据选择下一步。

## 执行与证据

OpenSpec：`add-copmrec-v4-collaborative-residual`。CPU等价/梯度/cold/checkpoint、Hydra/shell、严格spec、Mutagen与真实双卡smoke先于正式启动。W&B自动记录实际config/notes/source/used artifacts与验证best。SSH启动记录固定PID、启动时间、run id、外部log、terminal exit标记；观测超时先核对同一process，不盲目重启。

证据目录：[copmrec-v4-autonomous-20261005](evidence/copmrec-v4-autonomous-20261005)。当前状态以该目录和research-state记录为准，效果尚未验证。

2026-10-05实施与准备核验：163项CPU回归通过，Ruff/OpenSpec strict通过，Mutagen三会话Watching且无冲突，remote13关键文件SHA逐项匹配；v0 best48000实际SHA为567aed4610a2cfe6671a1eddd300470184d402853269559a3e719da95ba9a322，alpha0.8132556080818176。物理GPU2,4的双卡真实batch128 dry-run完成1更新，NCCL两进程、退出码0；本次不产生正式W&B run，检查脚本初次误用了world_size文本标记，已依据真实registered2processes/LOCAL_RANK日志纠正，未重复计算。

正式残差训练已在2026-10-05 03:22:38（UTC19:22:38）启动，PID3257442、远端job目录`logs/autonomous/copmrec_v4_residual_20261004T192238Z`。03:23:01实测该句柄存活、尚初始化输入，未观察到W&B run ID；stage训练已启动1/3，剩余2槽。下一步核对真正训练更新与实际W&B来源，然后完成匹配续训对照。未得到验证/Testing效果，目标保持active。

03:24:30已观察到真实更新step349，run [z9envq1n](https://wandb.ai/baymaxam/GRID/runs/z9envq1n) 为running，两卡均使用3831MiB且有持续利用率。实际配置核验符合准备的v0来源、weights-only、6000步、hybrid验证、batch128/GPU、LR/warmup；349个源码文件快照来源verified，source_sha256=40295a3739cd868b4b4aa25ee2ad1dc7137f92b2ec3f634b50bd7c2d137469b8。尚未发生首次1000步验证，训练loss不作为推荐收益。

03:38:10核验首个训练finished、terminal max_steps6000及进程exit0；6次hybrid全量验证best为5000（NDCG10=0.04882854968、Recall10=0.09457585961），6000略回落。真实best checkpoint Artifact `copmrec_beauty_v4_residual_train-checkpoint:v0`，digest a5848f4bde0d44660a30054a70157cfc，文件MD5 R0YHUEl+fuVqnsXXVe4jmQ==，内部global_step5000、学习alpha0.82357818、seen残差平均L2=1.53259、cold行严格全零。来源与所有字节核验通过，证据`training-residual.json`。349源码归档全量审计通过，未启动Testing，尚未证明10%收益。

03:39:30按预先计划启动无残差纯续训对照，物理GPU2,4、PID3318880、job目录`logs/autonomous/copmrec_v4_control_20261004T193930Z`。运行代码未变，Mutagen再次flush后三会话Watching，无冲突。训练已启动2/3（累计计划12000步），剩余1槽；等待实际对照指标后决策，不能用首个试验的验证上升单独归因为残差。

纯续训对照[5rlfx2np](https://wandb.ai/baymaxam/GRID/runs/5rlfx2np)已终态finished、exit0、实际6000步。验证best6000的NDCG10=0.047863930464、Recall10=0.092295311391；真实选定checkpoint为`wandb://baymaxam/GRID/5rlfx2np?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt`，digest1db5c6123a1ae7a2f5085f98f22d9136。1548928个残差元素均精确为0；两臂完整model config仅scale1/0不同，data相同、trainer除输出目录相同、source和v0来源相同。v4比各自best对照提升NDCG10 2.0153%、Recall10 2.4709%，六个相同步数均正向。证据`control-comparison.json`；尚非Testing结论，不能宣布10%目标。

用户再次明确排序和其他模块均在改进范围。根据新增协同表示的匹配验证正面证据，追加一次固定v4完整SID混合概率终排验证：同best5000、同Mass20∪cold33候选、checkpoint学得alpha，不扫权重、不新增head或训练。一次推理同时保存同候选content参考排序，比较覆盖和前排换位；完整SID分数为逐层混合概率的log之和，非item级线性混合。默认content行为保持，mixed仅推理/验证，cold使用相同完整目录条件分布。旧v0曾做过同规则的未晋级验证，此历史仅作来源边界，不拿来否决新checkpoint或宣称新规则。

本次明确新增2个完整evaluation单卡推理：固定LIGER checkpoint重算原始dense验证指标（历史DDP local/global均值略有差异，不能直接精确比较），以及v4固定mixed终排与同候选content配对比较。两者各自使用单GPU、同22363原始evaluation用户，验证决定是否晋级；Testing仍0次、训练仍2/3。此次验证预算不构成反复规则扫描授权，下一步以实际损益决定保留或停止该固定规则。

固定排序实现完成，221项新旧版本组合CPU回归、Ruff、两份OpenSpec strict及实际shell参数/Hydra验证通过。Mutagen flush退出0、三个会话Watching无冲突；remote40个关键文件SHA匹配、两个实际best checkpoint字节/step及catalog buffers核验一致。GPU2真实batch32的mixed dry-run一批退出0、进程组峰值1792MiB，关闭正式W&B与输出写入，仅作装配证明。证据`ranking-implementation-verification.json`、`inference-prelaunch.json`、`inference-smoke.json`。

2026-10-05 04:23:24/04:23:33已分别启动两个正式evaluation job：LIGER reference物理GPU4→local0，PID3467119、`logs/autonomous/copmrec_v4_liger_validation_20261004T202324Z`；v4 mixed物理GPU2→local0，PID3467907、`logs/autonomous/copmrec_v4_residual_mixed_validation_20261004T202333Z`。每项NPROC1、全量evaluation；正式推理计数2，其中Testing0。W&B run id/终态/原始输出与标签独立核验待完成，不能据启动或dry-run宣布收益。

实际LIGER验证run [5azn5vm0](https://wandb.ai/baymaxam/GRID/runs/5azn5vm0)已finished/exit0，完整evaluation22363用户及原始末商品标签、checkpoint/输入lineage、合法完整SID输出、349源码归档文件与40prelaunch文件全部通过。其全目录dense验证R10=0.08979117291955462（2008命中）、N10=0.04647035630072118；hybrid不是本比较基线。证据`validation-liger-validation-audit.json`。v4 mixed实际run [amwchjkc](https://wandb.ai/baymaxam/GRID/runs/amwchjkc)仍running，source同为349文件/ce78ce661aecfb4b2e1e40ce50738e6837743d5d95cf593eb6801ed7fc960305，GPU2/local0、batch32、固定mixed/chunk64与best5000均核验。

完整配对evaluation已审计通过，amwchjkc finished/exit0。固定LIGER dense R10/N10为.089791172920/.046470356301；v4 mixed为.092787193132/.048091188597，+3.34%/+3.49%；同候选content为.094575861915/.048828577391，+5.33%/+5.07%；v4完整目录dense为.095112462550/.049114561194，+5.93%/+5.69%。mixed相对同候选content新增82、损失122、净−40；R10配对差95%CI[-.003040736932,-.000581317355]，N10差CI[-.001353263022,-.000130186436]，均低于零。因此关闭固定mixed规则、不实施以mixed终排为前提的新增竞争loss；默认content保留。覆盖3100/22363不是mixed收益，当前dense、hybrid均尚未在evaluation达到10%。证据`validation-residual-mixed-validation-audit.json`。

依据evaluation预先选择两指标均最高的v4完整目录dense部署模式，准备首次单卡Testing。它仍使用v0派生的共同训练表示、三loss和学习alpha checkpoint；部署用完整目录content+collaborative表示排序，明确不声称生成候选或mixed终排带来这项增益。此次Testing只验证选定dense模式是否达到固定真实LIGER dense门槛，不以Testing改选hybrid或调参数。训练仍2/3、剩余1槽；正式Testing尚未启动，目标仍active。

首次Testing结果揭示前，预先固定唯一剩余训练槽的条件方案：若当前已选择dense候选没有完成原10%目标，将残差参数组LR从1e-3调为2e-3（multiplier20），主干LR1e-4、scale1、同v0best48000零初始化、同6000步/批次/三loss/learned alpha/weight_decay.035保持；验证选点对齐当前已选择的dense部署。依据是六个匹配验证点和全量dense的正面推荐效果，未宣称underfit，也不以残差范数或Testing错误样本决定LR。此干预同时改变AdamW的梯度更新与每步LR×weight_decay收缩，只归因残差参数组的联合优化设置，主干轨迹也会随三loss梯度变化。

旧臂best5000按hybrid选点，新臂按dense选点，因此best对比包含selection差异；固定比较新step5000 dense与当前旧step5000 dense（R10 .095112462550/N10 .049114561194），不写成best-vs-best仅LR不同。新臂仍按预先固定dense验证指标选择Testing checkpoint。若只有NLL/范数变化而推荐未改善，停止固定lr20设置，不扩展LR序列或重置预算。训练槽仍未消费，此方案在当前Testing前记录，以避免Testing调参。

2026-10-05 04:49:32首次固定dense Testing已启动：物理GPU2→local0/NPROC1/batch32，PID3554334，job目录`logs/autonomous/copmrec_v4_residual_dense_testing_20261004T204932Z`。实际shell/Hydra、40冻结源码hash、best5000字节、历史042139al Testing trace/原始输入匹配的只读prelaunch已通过；source仍ce78ce...960305。独立审计将从新的实际dense输出bundle与原始Testing末商品标签复算，并与固定真实LIGERdense配对。正式推理累计3（evaluation2、Testing1/3），训练累计2/3，目标尚待证明。

## 首次 dense Testing 独立结果

run [1edkvgk5](https://wandb.ai/baymaxam/GRID/runs/1edkvgk5) 已 finished，固定 PID3554334 的 job 退出码0。实际配置为物理GPU2→local0、单进程、batch32，使用 `z9envq1n` 的 evaluation-selected best5000；部署固定 dense/content，关闭 candidate trace。审计重新扫描 `data/beauty/testing` 全部原始 TFRecord，以每个用户的末商品构造完整四层 SID 标签，严格对齐22,363个用户和历史 LIGER dense `042139al`，不以 W&B summary 代替复算。

| 完整 Testing | Recall@10 | NDCG@10 | 命中数 |
|---|---:|---:|---:|
| 固定真实 LIGER dense | 0.07141260117157805 | 0.03639071015160814 | 1597 |
| v4 dense best5000 | 0.07516880561641998 | 0.03844148315032377 | 1681 |
| 相对提升 | 5.259862241703184% | 5.6354300044485495% | 净增84 |

固定用户配对 bootstrap（PCG64、seed42、2000次）的 Recall@10 差值为0.0037562044448419263，95% CI `[0.0017439520636766087, 0.005769574743996774]`；NDCG@10 差值为0.002050772998715627，95% CI `[0.0010894393295941937, 0.003070170210928486]`。新增命中314、损失230；共同命中中481例前移、442例后移。区间均为正，支持这一个 checkpoint 在当前 Testing 上有推荐增益；区间条件于固定 checkpoint 和用户，不能代表训练 seed 不确定性。

双指标相对提升10%的原目标未完成：命中数距离1757仍差76，NDCG@10距离0.04002978116676896仍差0.00158829801644519。未据 Testing 改选 hybrid 或切换 checkpoint；唯一剩余训练槽的条件方案已在首次 Testing 结果揭示前记录。本次记录不改变训练计数或剩余槽位，目标保持 active。既有 Testing 参与历史开发的边界继续保留，不将此 split 宣称为未触碰的独立确认集。

输出 SID 的完整合法性与每用户十个候选唯一性、baseline dense top-k 与 target rank 一致性、四个 catalog buffer、输入 Artifact、validation-selected checkpoint 字节全部核验通过；全部349个源码归档文件与40个 prelaunch 文件 SHA通过，source SHA为 `ce78ce661aecfb4b2e1e40ce50738e6837743d5d95cf593eb6801ed7fc960305`。完整数值、原始文件哈希和 lineage 见[Testing 独立审计](evidence/copmrec-v4-autonomous-20261005/testing-residual-dense-audit.json)。

历史格式兼容边界：`042139al` trace 的 keys 实际为 `torch.int32`，当前 writer 的 keys 为 `torch.int64`，shape均为 `[22363]`且唯一；标签、dense rank/top-k与 target row 为 int64。审计首次因强制历史 keys 为 int64而中止，核验原始字节后只修正审计器：保留历史 int32 keys SHA `c7f7eac3b485141b8b13e3a9f188a3d8078a7e1f0fab73351a85d10dfee1884a`，将双方 keys 转为 int64后逐整数严格对齐，规范化 SHA同为 `5c14775a3aec5907a79e94c4cc0b2a00be6344c0e72cb2e40cca75e9c34744d9`。没有改动历史产物、推理源码或重新推理。

Testing 标签 SHA固定为 `ea9747417e5fec34daba59547f9e12fe14fea08c9c9ff2c512ea8944ae1e7939`；evaluation 与 Testing 的用户 keys相同，但 evaluation 标签 SHA为 `efe624caa09a26812661a9930c29724a5c0d23670c304fe5c2e48de56471a72f`。fresh raw末商品扫描与标签指纹共同防止 split 混用；历史源码 provenance缺失的边界也未回填。格式诊断见[baseline tensor format](evidence/copmrec-v4-autonomous-20261005/testing-baseline-tensor-format.json)。

## 唯一第三训练槽：预承诺 lr20

2026-10-05 05:00:11（UTC21:00:11）按首次 Testing 前记录的条件方案启动 residual-lr20，固定 PID3592927、job目录 `logs/autonomous/copmrec_v4_residual_lr20_20261004T210011Z`。物理GPU2,4→local0,1、双进程、每卡batch128/global256；仍从同v0 best48000 weights-only开始，新增残差全零，主干LR1e-4、残差LR0.002/multiplier20、scale1、三loss各1、learned alpha、6000步。验证与预测均固定 dense，最终分数 content+共享协同残差。启动依据为已登记的 Validation 正面证据与条件触发，未用新的 Testing 错误样本选择LR或训练步数。

实际shell参数与Hydra、Mutagen flush及三个Watching、55个运行文件SHA、v0 checkpoint字节/step、两张卡空闲已由root在启动前核验。模型结构、三loss与DDP批次均未变，复用已通过的双卡一更新smoke及221项default回归；本次没有修改runtime源码。expected source仍为 `ce78ce661aecfb4b2e1e40ce50738e6837743d5d95cf593eb6801ed7fc960305`，实际W&B来源及终态由同一job观测确认，不以启动证明效果。

累计训练已启动3/3、剩余0，已承诺总步数上限18000；尚不能把第三臂的6000步计为已完成。Testing仍1/3、剩余2，未自行追加Testing。终态须审计六次dense evaluation、实际validation-selected best Artifact与字节，并固定比较新step5000 dense对旧step5000 dense（R10=0.09511246254974735、N10=0.04911456119404004）。旧best按hybrid选点、新best按dense选点的差异保留，不能把best对比宣称为仅LR的因果效果。若固定lr20未改善，停止该设置，不重置训练预算或开启LR序列；全线程10%目标仍未完成。

05:05:12（UTC21:05:12）实际同job观测 run [nj9elah1](https://wandb.ai/baymaxam/GRID/runs/nj9elah1) 为running、step2199；已完成1000/2000两次dense验证，最新step2000 R10=0.09140097349882126、N10=0.04724394157528877。W&B实际配置符合dense/content、multiplier20、baseLR1e-4、warmup300/max6000、同v0 weights-only来源；GPU2,4映射LOCAL_RANK0/1，均约3831MiB且有持续利用率。source record为349文件/ce78ce...960305，完整归档与残差训练契约、optimizer/scheduler两组LR留待终态只读审计，不据中途loss或两个验证点宣布目标完成。

05:12:37（UTC21:12:37）同一训练句柄确认 exit0、W&B finished、实际完成6000步；第三臂的六次 dense 验证如下，validation-selected best 为6000。checkpoint 内部 global_step=6000 与 history 最后一条 trainer/global_step=5999 的更新计数边界一致。

| 更新步数 | Recall@10 | NDCG@10 |
| --- | ---: | ---: |
| 1000 | 0.08979117125272751 | 0.046470757573843 |
| 2000 | 0.09140097349882126 | 0.04724394157528877 |
| 3000 | 0.09363681077957153 | 0.048232946544885635 |
| 4000 | 0.09479944407939911 | 0.0490557923913002 |
| 5000 | 0.09560434520244598 | 0.049646779894828796 |
| 6000 | 0.09600679576396942 | 0.04979391396045685 |

真实 best URI 为 `wandb://baymaxam/GRID/nj9elah1?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt`，Artifact `copmrec_beauty_v4_residual_lr20_train-checkpoint:v0`、digest `37c60160df1e0f767728526644aed7dc`；文件182848285字节、SHA256 `7c52228a665b593b50e45e8aba039f3f4aa4a10ca134d9345c59c59fb142f055`。checkpoint 中 alpha=0.8296189904212952、seen 残差平均L2=2.646622657775879，cold 行仍严格全零。checkpoint 的 `copmrec_collaborative_residual_training.residual_lr_multiplier` 明确为20，W&B配置、optimizer initial_lr及scheduler base_lrs均为主干0.0001/残差0.002，两组weight_decay=0.035。选中6000步的当前调度LR均为0，符合预定cosine终点；这一值与训练配置中的初始LR分别记录。

补充审计同时核验训练来源v0 best48000、SID与embedding实际 used_artifacts、物理GPU2,4及LOCAL_RANK0,1、完整349源码文件与55个prelaunch SHA；runtime源码仍为 `ce78ce661aecfb4b2e1e40ce50738e6837743d5d95cf593eb6801ed7fc960305`，没有新增运行代码。固定step5000 dense对旧v4同step5000 dense的标量比较为R10 +0.5171589921156405%、N10 +1.083627111491614%；此时尚无原始输出配对置信区间，不把标量差异或两个best的选择规则差异当作纯LR因果效果。证据为[训练best审计](evidence/copmrec-v4-autonomous-20261005/training-residual-lr20.json)和[训练契约与源码补充审计](evidence/copmrec-v4-autonomous-20261005/training-residual-lr20-supplement.json)。

累计3/3训练均已终态，实际完成总18000步，训练剩余0；Testing保持1/3、剩余2，全线程10%目标仍未完成。

## lr20 best6000 完整 Validation 确认

05:14:55（UTC21:14:55）root启动唯一固定 lr20 best6000 完整evaluation推理：物理GPU2→local0、NPROC1、batch32，PID3647439，job目录 `logs/autonomous/copmrec_v4_residual_lr20_validation_20261004T211455Z`。配置为 `prediction_mode=dense`、`final_ranking_mode=content`、`candidate_trace=false`、trace writer=null，明确 `data/beauty/evaluation`；使用上述真实validation-selected best6000与同SID/embedding，运行源保持冻结。启动前Mutagen flush及三个Watching、40个推理文件SHA与55个训练文件SHA核验通过。

05:16:05（UTC21:16:05）固定句柄观察确认进程已退出、exit0，实际run [mq8hhof5](https://wandb.ai/baymaxam/GRID/runs/mq8hhof5) finished，W&B报告22363用户，实际resolved config与预定prediction/split/checkpoint一致，source record为349文件/同ce78源码。其summary暂报R10=0.09600679694137638、N10=0.049793932567196414；这里只记录运行终态，完整原始evaluation用户/末商品标签、合法唯一SID输出、source与输入lineage、固定LIGER和旧v4 dense配对CI待独立审计，暂不作为正式晋级结论。观测证据为[固定句柄终态](evidence/copmrec-v4-autonomous-20261005/observation-residual-lr20-validation-latest.json)。正式推理累计4次，其中Validation3次、Testing1次；本次未启动Testing或改变已选择的dense模式。

05:23:31（UTC21:23:31）独立原始审计完成、exit0，重算结果如下。175个原始evaluation文件、22363个用户与末商品标签完全对齐；新输出每用户10个完整SID均合法且唯一。固定LIGER和旧v4参考仅使用真实全目录dense rank/topk，未把hybrid或mixed指标作为基线。

| 完整evaluation，固定checkpoint | Top10命中 | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| LIGER best45000，5azn5vm0 dense | 2008 | 0.08979117291955462 | 0.04647035630072118 |
| v4 lr10 best5000，amwchjkc dense | 2127 | 0.09511246254974735 | 0.04911456119404004 |
| v4 lr20 best6000，mq8hhof5 dense | 2147 | 0.09600679694137638 | 0.049793932567196456 |

lr20相对LIGER的R10 +6.9223107569721165%、N10 +7.152035256557077%；新增命中477、损失338、净139。配对差值95%CI分别为R10 `[0.0037562044448419263, 0.008676161516791122]`、N10 `[0.002070929775772703, 0.004585037020978065]`，均为正，但两项点估计仍未达相对10%。相对旧v4dense的R10 +0.9402914903620108%、N10 +1.3832382019506984%，新增159、损失139、净20；两项CI分别为 `[-0.000536600634977418, 0.00236998613781693]`、`[-0.000015168409071523037, 0.0013225257441768085]`，均跨0，不能宣称lr20已经确证优于旧v4。bootstrap为固定checkpoint条件下2000次配对用户抽样、seed42、未调整的逐项CI；新旧best选点规则不同的边界继续保留。

审计确认推理源码349文件+40个prelaunch hash、训练源码349文件+55个prelaunch hash、真实best6000字节/step/SHA、v0/SID/embedding输入以及GPU2/local0单进程日志均通过；输出文件SHA为 `aaa99dd8cffc2965c491baccb057558a4e8ca4cc7fa3b2075cce2052eb164b02`。历史v0 checkpoint未包含后加的version/alpha_policy字段，审计遵循冻结的 `initialize_from_v0` 默认v0/learned，同时严格核对joint标识、没有fixed alpha及完整state keys，显式记录该旧schema兼容边界，没有补写旧Artifact。证据为[lr20独立Validation审计](evidence/copmrec-v4-autonomous-20261005/validation-residual-lr20-audit.json)。本轮不消耗Testing，不启动新forward，不改变累计预算或全线程目标状态。

同一审计另存现成dense Top10的描述性曝光统计：旧v4的20个最高false曝光商品占false Top10曝光22.45296835899353%，lr20各自top20占18.50932091474533%；冻结旧20商品在新输出的false曝光由49734降至39475（减少10259），新旧top20集合重合16/20。false曝光指某用户Top10中的单个商品SID不同于该用户原始末商品标签；该统计只描述已有输出，不把曝光集中、集中度下降或旧商品集合当作因果效应，也未拟合任何bias。evaluation仍是development数据，不能据此代替Testing目标验证。

## 本阶段关闭决定：残差正向但未达全线程目标

2026-10-05 05:27:07（UTC21:27:07）正式关闭该v4自主残差阶段，状态为 `closed_budget_exhausted_positive_residual_below_goal`。3次训练全部结束、实际18000步、训练槽剩余0；推理共4次，其中Validation3次、Testing1次。阶段不再追加残差训练、LR试验序列、固定mixed终排或mixed竞争loss，不重置预算。尚未使用的2个Testing槽保持未消费；本阶段没有继续使用它们的授权。

保留协同残差的正向推荐证据：原两臂匹配6000步的六个等步验证点均正向，正式单卡dense Testing `1edkvgk5` 使用 `z9envq1n` best5000，R10/N10相对固定真实LIGER dense分别+5.259862241703184%/+5.6354300044485495%，两项配对CI均为正。这是当前已验证deploy结果，仍未达到全线程两指标相对+10%的目标；不将新best6000的Validation数字替换为Testing结果。

关闭固定mixed规则：同候选content比较的R10与N10配对CI均为负。第三训练槽lr20已完成，并将 `nj9elah1` 的dense-validation best6000保留为开发起点；该checkpoint的完整Validation相对LIGER为+6.9223107569721165%/+7.152035256557077%，仍不足10%。相对旧v4dense的两项配对CI均跨0，额外LR收益未确证；固定step5000两指标为正只作为标量验证记录，保留缺少配对CI以及新旧best选点规则不同的边界，不宣称纯LR因果改善。因此新best6000不进入第二次Testing，Testing尚未做。

这一关闭决定只结束当前预算下的残差/LR/mixed阶段，全线程目标保持active且 `target_achieved=false`。下一问题暂限于score bias只读证据与当前checkpoint固定半组统计，由新的五项路线门禁判断是否值得另立阶段；当前没有授权新的stage launch、模型/配置/脚本改动、训练或推理。已有曝光描述不能直接证明bias机制有效，不能作为自动扩预算的理由；新的问题、预测、必要对照与累计预算须单独登记后才能运行。
