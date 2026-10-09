# BRIR 有限审计与 Sports 内容初始化结果

后续状态：Sports独立evaluation推理已完成。全量指标、Head/Mid/Tail配对、不确定性与下一步判断见[Sports结果报告](sports-content-init-outcome.md)。下文保留训练期与BRIR审计的原始分析。

日期：2026-09-16。来源：W&B `baymaxam/GRID` 的本轮5个正式run、4份真实audit v2产物、5个Sports训练条件的完整验证历史及checkpoint metadata。所有本轮run均finished。本次只读取、下载和分析证据，没有启动训练、推理或修改远端run。

## 1. 决策

停止将BRIR v1作为主方法继续扩展。审计排除了本批用户上的动态搜索遗漏，揭示了共同候选目标与全目录竞争之间的偏移：候选损失改善不保证全目录排序改善。两个残差分支大量使用饱和分数压低冻结hard negatives，同时提高其余目录分数，没有恢复任何新的Top10命中。

Sports内容初始化在匹配的20k、seed42验证协议下超过CGBS full。结合既有Beauty结果，简单内容初始化应成为后续研发必须超过的基线；复杂前缀/多原型结构的必要性尚未建立。该基线本身不是已确定的新论文贡献，也尚无Sports长尾效果结论。

## 2. 身份与证据有效性

| 本轮任务 | run | 来源/选择 | 结果 |
|---|---|---|---|
| BRIR base audit | [mwd1gls1](https://wandb.ai/baymaxam/GRID/runs/mwd1gls1) | ffa9bc2e last，20k | finished，128行 |
| dense audit | [vzdpqysz](https://wandb.ai/baymaxam/GRID/runs/vzdpqysz) | iv1h5zky last，5k | finished，128行 |
| prefix_free audit | [he83qop6](https://wandb.ai/baymaxam/GRID/runs/he83qop6) | 2ux40teb last，5k | finished，128行 |
| brir audit | [tvq4crwp](https://wandb.ai/baymaxam/GRID/runs/tvq4crwp) | 82t0rv4c last，5k | finished，128行 |
| Sports token_content_init | [6x3rdo7k](https://wandb.ai/baymaxam/GRID/runs/6x3rdo7k) | 从头训练20k，best为18k | finished，40次验证 |

四份audit均通过当前v2校验器。逐行keys、labels、输入摘要、目标item及81个候选keys完全一致；四份全目录anchor分数最大绝对差为0。目录含12,101个item。

- 公共catalog指纹：`c095505f44e77e9125397123d321a3b4029511feeff9f09ec0197d5dfb8d0b13`。
- 公共anchor指纹：`62c7a0531b0c4efd228a1527ec0b7f707295801cc40419a901524ebf58b5f953`。
- 四个checkpoint指纹与此前已下载的实际checkpoint state_dict重新计算结果一致，实际global_step分别20k/5k/5k/5k。
- Beauty上游为SID v2、embedding v5；三个分支还共同使用base last和py8ovqwa标定v0。evaluation，抽样seed42，候选seed42，GPU0。
- Sports五个条件上游均为SID v3、embedding v6；seed42、GPU devices=[0,1]、每卡batch128、20k steps、500步验证、32-true、从头训练、无test选模。新条件与mask_ce的encoder、decoder、训练组件配置、SID深度、序列长度等核对一致。
- Sports最佳checkpoint metadata与完整history峰值一致（旧run JSON数值在约1e-18量级有序列化差异）。本次没有下载/恢复新的Sports checkpoint tensor。

## 3. BRIR：搜索正确，评分与优化目标仍失败

以下只代表同一批128个evaluation用户，不是全量指标。候选CE是在这些evaluation历史上按原训练候选规则重建的目标+64 hard+16 random集合，不能冒称保存了真实训练batch。所有CE为eval模式原始值，无梯度累积缩放。

| arm | Top10命中/128 | NDCG@10 | 候选CE | 全目录CE | 候选外概率质量 | 每用户Top10候选外数量 |
|---|---:|---:|---:|---:|---:|---:|
| base | 9 | 0.025689 | 6.048402 | 8.177028 | 83.458% | 0.000 |
| dense | 0 | 0.000000 | 3.820044 | 9.709966 | 99.696% | 9.984 |
| prefix_free | 8 | 0.023337 | 5.357789 | 8.242480 | 91.439% | 1.711 |
| brir | 7 | 0.021078 | 5.359157 | 8.241218 | 91.448% | 2.180 |

### 3.1 搜索并未隐藏更好的评分结果

每个arm的dynamic与reference均为128/128严格有序Top10一致，原始bound均成立。BRIR dynamic平均评价114.25个残差item，reference评价12,101个；扩展到全目录没有增加命中。

fixed64/128并非精确：BRIR有序Top10一致率分别49/128、106/128，只是这批样本恰好得到相同目标命中/NDCG。不能据此宣称固定候选搜索等价。BRIR fixed256为128/128一致；prefix_free fixed256仍有3个用户不一致。

这里的item计数是残差搜索计数，基础dense评分本身已经扫描全目录。reference逐块评分循环有实现开销，不能把12,101/114.25或计时比直接称为端到端加速。

### 3.2 Dense 的失败与固定候选支持集偏移一致

相对冻结anchor：104/128用户候选CE改善，89/128用户全目录CE恶化，其中65个用户同时发生两者；93/128目标排名恶化，原有9个命中全部丢失。平均目标排名2046.21→5536.58。

其最终Top10的1280个推荐位置中，有1278个不在重建候选集内。当前Top64 hard negatives与冻结Top64的重合，在128×64个位置中仅1个。

平均分数变化：目标+3.375、冻结hard negatives +0.042、随机负例+5.514、候选外+5.542。目标相对冻结hard negatives改善，但相对大量其余item变差。候选外概率质量本来就可能很高，关键证据是同query前后变化、目标排名和候选内外竞争共同恶化，不是两种CE的绝对差值。

这支持当前固定候选训练方案发生目标偏移，不能用它否定所有dense retrieval、item CE或联合训练。尚未通过更换训练目标的干预实验排除所有其他训练因素。

### 3.3 残差几乎饱和，修正方向没有转化为命中

两分支delta均为0.4855576195。以绝对残差达到0.95×delta定义饱和，统计覆盖128×12,101个query-item分数：

| 统计 | prefix_free | brir |
|---|---:|---:|
| 饱和比例 | 98.732% | 98.687% |
| 正向饱和比例 | 97.763% | 97.617% |
| 目标平均残差 | +0.2555 | +0.2505 |
| 冻结hard negatives平均残差 | −0.4503 | −0.4577 |
| 随机负例平均残差 | +0.4735 | +0.4713 |
| 候选外平均残差 | +0.4758 | +0.4746 |
| 相对base新增/丢失Top10命中 | 0 / 1 | 0 / 2 |

这与压低冻结hard negatives、普遍抬高其余目录的行为一致，未形成足够有效的个性化item修正。大量正向公共偏移本身不能改善排序。这里描述的是评分行为；饱和并不能单独证明因果根源。

BRIR确实改善了66个目标的全目录排名，恶化23个，其余39个相同，平均排名2046.21→2032.90；但没有一个新的Top10命中，且丢失2个原命中。不能将中后排的局部改善等同终端推荐改善。

### 3.4 幅度界有限，但不是唯一解释

在当前base分数和delta下，允许目标获得+delta、所有竞争item获得−delta的逐query乐观排名表明：30/128用户理论上可以进入Top10，包含base已有9个和额外21个。另98个用户即使理想独立残差也无法进入Top10。

因此当前界限制了可恢复范围；同时可恢复的21个用户实际一个也未救回，说明不能只靠“增大delta”解释或解决本轮失败。该乐观界允许每个query独立选择残差，不代表共享网络一定可达。

## 4. Sports：简单初始化超过复杂内容方案

下表全部来自训练期evaluation验证，按NDCG@10选择同一checkpoint，Recall取同一步。五个best恰好都为18k；不混用既有独立推理指标。

| arm | run | best NDCG@10 | 同步Recall@10 | 最后5次验证NDCG均值 |
|---|---|---:|---:|---:|
| original | gsn7uqub | 0.016722 | 0.032250 | 0.015892 |
| mask_ce | lhevwroq | 0.018552 | 0.034329 | 0.017988 |
| hybrid | 8fzuuf9v | 0.020170 | 0.037615 | 0.019121 |
| full | bpt1zqo9 | 0.020540 | 0.038514 | 0.019682 |
| token_content_init | 6x3rdo7k | **0.021426** | **0.039778** | **0.020498** |

内容初始化best NDCG相对mask_ce +15.49%、hybrid +6.23%、full +4.31%。最后20k的NDCG为0.020044，高于full的0.019524；最后5次均值也领先。40次验证中22次超过full，后20次中14次超过，优势并非每个时间点都成立。连续检查点高度相关，不能充当独立seed或显著性证据。

新的训练耗时2442秒，约40分42秒。虽然结构简单，本轮并未观察到相对full（2346秒）的更短墙钟时间，不能宣称已证明训练加速。

尚未发现新Sports内容初始化的独立inference/diagnosis run，因此目前无法判断它的增益来自Head、Mid还是Tail。既有Beauty简单初始化优于full的结论见research仓库CGBS outcome报告；新Sports仅在当前训练验证协议下补上同向证据。

## 5. 下一步与论文主张边界

1. BRIR v1结束扩展，不追加它的seed、数据集或delta扫参。保留搜索界和审计基础设施作为可复用工具。
2. 以内容初始化作为强基线开展后续设计。CGBS full在Beauty和新Sports结果中均未建立对该简单方案的结构优势，不能继续以复杂度本身支撑创新。
3. 先完成Sports内容初始化的同协议keyed evaluation，随后连接已有训练频次分组做配对Head/Mid/Tail分析。若增益仍主要来自Head，需要收缩原长尾假设。
4. 新主方法需要针对已验证的具体评分/完整路径竞争缺口，明确目标、干预和失败门槛。本轮支持“内容信息有效、当前固定候选残差目标存在问题”，不支持“新颖主方法已确定”或“长尾问题已解决”。

无需为以上判断重新训练。下面仅为下一步手动推理命令，本次未执行；选择已发布的18k best checkpoint，evaluation，单GPU0，beam10，保留prefix trace：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash ./tiger_catalog_grounded_inference.sh \
  --data-dir data/sports \
  --dataset sports \
  --group rkmeans \
  --semantic-id-path wandb://3narllqy \
  --embedding-path wandb://psec3u5i \
  --checkpoint-path 'wandb://6x3rdo7k?role=checkpoint' \
  --data-split evaluation \
  --devices '[0]' \
  --arm token_content_init \
  --beam-width 10 \
  --seed 42 \
  --notes "Sports content-init best18k evaluation; matched CGBS beam10; keyed outputs and prefix trace; no new training"
```

## 6. 可复核文件

- [原始run快照](../../../tmp/brir_closure_results/runs.json)、[完整history与Artifact lineage](../../../tmp/brir_closure_results/evidence.json)。
- [校验与统计结果](../../../tmp/brir_closure_results/analysis.json)、[512行逐用户指标](../../../tmp/brir_closure_results/per_user.csv)、[候选内外分数变化](../../../tmp/brir_closure_results/score_change_details.json)。
- [对比图](../../../tmp/brir_closure_results/overview.png)。
- 四份原始 `brir_audit.pt` 位于上述分析目录下各自run ID子目录。

未自动启动完整GPU任务，未更改模型/配置/实验脚本，未提交代码。128用户审计只支持本批机制诊断，Sports单seed验证仍需推理复核与后续独立重复，不能作为论文最终测试结论。
