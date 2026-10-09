# CoPMRec v5.3 固定验证结题（2026-10-06）

## 决定与结果

唯一授权的新增从头50k训练与单卡完整Validation均已完成，训练及原始输出主审计、附加独立复核通过。v5.3相对固定同seed、同更新预算LIGER dense有明确部分推荐收益，但Recall@10未达到+8%；相对v5.2的增量尚不确定。按实施前proposal的决定分支，停止这项固定辅助CE干预，保留v5.2主方案，同时保留v5.3 checkpoint、代码与相对native的部分收益证据。不自动扫描权重、改门槛或启动新实验。

训练run：[hqw189d2](https://wandb.ai/baymaxam/GRID/runs/hqw189d2)；完整Validation run：[6u6g62hk](https://wandb.ai/baymaxam/GRID/runs/6u6g62hk)。

| 方法 | Recall@10 | NDCG@10 | 相对native R10 | 相对native N10 |
| --- | ---: | ---: | ---: | ---: |
| LIGER dense（native42，固定主对照） | 0.0969458481 | 0.0540488986 | — | — |
| v5.2（冻结描述性对照） | 0.1014622367 | 0.0610582814 | +4.6587% | +12.9686% |
| v5.3 | 0.1037875061 | 0.0615505877 | +7.0572% | +13.8794% |

配对bootstrap为PCG64 seed42、2000次重采样；区间是指标绝对差的pointwise 95% CI，不能解释为相对收益百分比区间。

| 比较 | R10相对增量 | R10绝对差95% CI | N10相对增量 | N10绝对差95% CI |
| --- | ---: | --- | ---: | --- |
| v5.3 − native42 | +7.0572% | [0.0033079193, 0.0105095470] | +13.8794% | [0.0051864509, 0.0097791378] |
| v5.3 − v5.2 | +2.2918% | [-0.0002235836, 0.0048294057] | +0.8063% | [-0.0008383727, 0.0018169660] |

v5.3相对native的两项区间下界均为正，但R10点估计不足+8%，因此预定“双8%且两绝对CI下界正”门禁未通过。2342个命中才满足固定R10点估计门槛，实际2321个，差21个；这只是算术差额，不是可恢复用户数量或新实验效果估计。所有相对v5.2的@5、@10增量区间均跨0，不能把这一次最高点估计称为确定的辅助CE增益。

## bad case：恢复与破坏同时存在

仅重读三份已经固定的Top10输出、SID与175份Evaluation；没有新模型forward、checkpoint评分、Top11、过滤补位或新预测。四组由native42和v5.2的命中关系预先定义，未按v5.3结果筛组选样。

| 固定用户组 | 用户数 | v5.3命中 | 对v5.2的作用 |
| --- | ---: | ---: | --- |
| A：native与v5.2均命中 | 1344 | 1249 | 丢失95 |
| B：native命中、v5.2遗漏 | 824 | 149 | 恢复149，仍遗漏675 |
| C：native遗漏、v5.2命中 | 925 | 645 | 保住645，丢失280 |
| D：native与v5.2均遗漏 | 19270 | 278 | 新增278 |

相对v5.2新增命中427（B149＋D278）、丢失375（A95＋C280），净增52。恢复方向并非完全无效，但同时破坏了部分既有覆盖，净增远小于输出变化规模。这个分解描述预测集合变化，不能证明辅助CE是每一组变化的单独原因。

相对v5.2，共同命中的用户中557名次上升、535下降；共同命中名次变化对整体N10差的贡献为−0.0001540930。N10净变化由新增命中贡献+0.0075410655、丢失命中贡献−0.0068946662与共同命中位置贡献相加得到+0.0004923063。结果支持“覆盖恢复伴随覆盖损失”的描述，未证明已有命中的排序质量得到可靠提升。

相对native则新增923、丢失770、净增153，共同命中590上升、437下降。全部2321个命中来自seen目标；51名cold目标用户仍为0命中。seen用户错误Top10中的cold占位由v5.2的44降至10，仅描述当前输出，不能从Top10推断补位收益，也不能据此推出cold能力。

后续若重新立项，问题应收敛为如何在恢复B组或D组覆盖时保护A组与C组既有命中；本次没有证明某个新的保护机制，当前不追加方法、预算或实验。

## 方法、预算与证据边界

v5.3只在v5.2上增加权重1.0的全目录native-view CE，训练共享query及同一次projection/dropout，部署继续用原联合分数；没有新增参数。native-view只去掉目录侧残差，history/query仍共享残差路径；初始化残差为0时两个内容CE相同，内容监督总量增加。因此单臂结果无法区分视图监督作用与增加内容loss权重作用，也不能把描述性bad case当作因果隔离。

随机初始化、全部模块共同连续50000更新，无teacher、外部推荐CP、分段续训或optimizer重启。物理GPU3、6→CUDA_VISIBLE_DEVICES=3,6→local[0,1]训练，单物理GPU6→local[0]推理；global batch256、FP32、AdamW、原学习率和50k scheduler保持。own-best45000由100次raw Validation的首个N10最大值选择；165个state tensors、154份optimizer moments与scheduler完整保存至45000。没有50000终态完整状态checkpoint，实际50000训练预算由完整历史及正常终止独立证明，不能把45000保存状态改称50000。

使用checkpoint：

```text
wandb://baymaxam/GRID/hqw189d2?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=045000.ckpt
SHA256: 29b9794d3a9079e63173457c894c3cdb9f8ab1de9a4e16d76bf68f4050e11e62
```

原始Validation核验175文件、22363用户、标签/历史/用户身份、合法无重复Top10和实际checkpoint/输出Artifact lineage；采用与对照相同的完整目录历史资格策略。source archive实际核验408文件，SHA256 `d6c5000509d968caabf5b7d46b5c5be3889c0c9b5e857d60fe5cb1d4cf3b22a0`；原v5.2的402运行文件字节不变。训练50k与native同更新预算，不声称相同FLOPs、GPU时间、HPO成本或证明native已经收敛。

本次消耗新增1train/50k＋1完整Val。累计当前统一阶段4train/200000更新、4完整Val、启动attempt5（此前预测前失败1次），新Testing0、seed43 pair0；旧阶段账本与研究状态闭合尾部原字节保持。seed42开发集已反复使用；native42历史训练源码缺口不由新source补写，单seed和pointwise CI不能代替跨seed复现与Testing。整体可复现双8目标仍未完成，新增额度已用完。

## 实际凭据

- [训练主审计](evidence/copmrec-unified-native-view-ce-50k-20261006/training-candidate42.json)与[独立终态复核](evidence/copmrec-unified-native-view-ce-50k-20261006/training-independent-review.json)。
- [完整Validation主审计](evidence/copmrec-unified-native-view-ce-50k-20261006/inference-val-candidate42.json)、[附加独立复核](evidence/copmrec-unified-native-view-ce-50k-20261006/validation-independent-review.json)及[固定四组分析](evidence/copmrec-unified-native-view-ce-50k-20261006/quad-cohort-existing-output-analysis.json)。
- [原始授权注册](evidence/copmrec-unified-native-view-ce-50k-20261006/stage-registration.json)、[累计实际账本](evidence/copmrec-unified-native-view-ce-50k-20261006/cumulative-budget-latest.json)及[实施前固定proposal](copmrec-v5-3-native-view-ce-proposal-20261006.md)。

核心74项、配置/脚本40项聚焦验证及独立review此前通过；此次正式训练、单卡完整Validation与全部已约定审计均实际完成。阶段闭合helper独立复核通过，root实际登记成功（receipt SHA256 `4175c6830c2e39845db2a5e7c2626e99cce0b4b838f3aae33e5786a90f3dd6be`），旧闭合尾部及父账本保持；OpenSpec strict通过。
