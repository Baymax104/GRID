# CoPMRec 连续50k整体模型：v5.1结果与路线门禁

## 当前决定与证据范围

保留 v5 的已确认 NDCG／Top5 部分收益，停止 v5.1 固定双目录平均及其权重扫描。整体随机初始化、连续50000更新、相对同预算 LIGER dense 双指标至少8%及成对复现目标仍未完成，goal 保持 active。停止对象是本次具体评分干预，不是整个共享协同残差模型。

v5.1实际训练 [cm0i584p](https://wandb.ai/baymaxam/GRID/runs/cm0i584p) finished／exit0，完整100个raw Validation点证明50000更新；自身raw NDCG@10最佳点为46000，单卡完整Validation [w19eutzj](https://wandb.ai/baymaxam/GRID/runs/w19eutzj) finished／exit0。175原始Evaluation文件、22363用户、标签、输出合法性／唯一性／有效history排除、checkpoint和396文件实际源码归档已独立核验。best／last的完整训练状态均保存至46000，不冒称有50000步完整状态文件。

| 完整Validation指标 | 匹配native42 | 保留v5 | v5.1 |
|---|---:|---:|---:|
| Recall@5 | 0.06515226043017484 | 0.07239636900236998 | 0.06779054688548049 |
| NDCG@5 | 0.0438338045937098 | 0.051138330645159706 | 0.04696004567618626 |
| Recall@10 | 0.09694584805258687 | 0.10043375217994008 | 0.09538076286723605 |
| NDCG@10 | 0.05404889855718787 | 0.06015084675195483 | 0.05584562468008188 |

v5.1相对native42的R10−1.614391%、N10+3.324260%，配对绝对差CI95分别为[-0.0050529893,0.0017886688]及[-0.0003384832,0.0039061625]，均跨0，双8%门禁false。相对冻结v5，R10−5.031167%、N10−7.157376%，两CI上界均为负，Top5两指标也下降。native42仍是原整体门禁唯一分母；v5比较用于判断具体干预，不替换分母。统计条件限定于固定checkpoint和既有开发使用的Beauty Validation，未构成独立Testing或第二seed验收。

## 五项路线门禁

### 1. 核心可反驳假设

阶段问题保持为：把已保留协同残差及内容建模收益转为从step0共同训练、单checkpoint部署的50k整体模型，同时改善覆盖与排名。v5.1的具体假设是，含残差的唯一history query对两个目录的固定0.5／0.5 cosine logits平均，可以相对v5增加Top10净命中并保留头部收益。

### 2. 正面证据及其支持强度

v5完整Validation的N10相对native42+11.289681%，配对CI下界为正；Top5也改善，支持保留共同训练的部分收益。R10仅+3.597786%且CI跨0，不能当作已确认覆盖增益。用户已接受的旧64k预算匹配整体约8%结果保留，但它使用续训与两个独立query的checkpoint融合，不能证明随机初始化单模型50k效果，更不能证明共享query双目录干预有效。

### 3. 反面、未过门槛与缺失证据

v5.1有377新增／490丢失Top10命中，净−113；相对v5，Top5净−103、6–10桶净−10，共同命中的NDCG位置贡献也为负。两个预固定key组相对v5的R10／N10均下降且CI为负。因此，本次具体“增加覆盖并保留头部”的预测不获支持，应停止这版固定评分干预。它不排除其它协同残差机制，也不证明所有双视图方案均无效。

v5和v5.1的整体双8%门禁都未通过；第二seed成对复现和新Testing均未运行。NDCG点增量3.32%但CI跨0、R10下降，不应将v5.1包装成新的有效组件。

### 4. 当前决策需要的最少鉴别

训练配置、来源、预算、cold残差零、CP恢复与完整输出有效性已核，没有需要重跑训练来解决的实现疑点。v5.1只改评分代码，但梯度仍更新共享projection、history residual及T5，最终Top10不能区分query和目录的原因；不为穷尽这些原因追加实验。

已有输出出现一个具体线索：false cold Top10占位从v5的6638增加至v5.1的11105；seen目标用户内为6614→11079。cold目标仅51用户，命中6→8，不能推广冷启动效果。对于非退化的投影向量，平均单位目录向量在seen商品上范数不超过1，而cold残差为0时两路一致、范数为1；这是函数代数，不能证明增加占位造成113次净损失。

只读核对native42／v5／v5.1现有Top10、已存CP目录与原标签，记录cold占位用户数、cold在已命中seen目标前的数量，以及新增／丢失／共同命中组分布。不生成新分数或推荐列表，不扫描参数，不估算Top11恢复。它只回答最后一个训练槽是否有具体资格／校准问题值得鉴别；若没有足够支持，关闭该疑问，不把未知原因自动变成新模块。

### 5. 累计预算及下一步

原阶段上限保持3次正式训练／150000更新、3项完整Validation、3次新Testing，不因版本名重置。实际已完成2次训练／100000更新、2项完整Validation；Validation正式启动attempt3，其中1次旧startup在预测前失败，单独保留。新Testing0。剩余一个50000训练槽及一项完整Validation尚未分配。

当前仅安排上述已有输出只读鉴别，新增训练／完整推理为0。不得自动开展seed43／Testing或扩大预算。原双seed条件保持；剩余一个训练槽不足以覆盖native43与candidate43两次成对训练。后续任何训练分配须先登记具体问题、正面依据、固定方法、成本与正／负／不确定结果触发的决定。

## 可审计来源

- [实际训练审计](evidence/copmrec-unified-dualview-50k-20261005/training-candidate42.json)与[训练独立复核](evidence/copmrec-unified-dualview-50k-20261005/training-independent-review.json)。
- [完整Validation原始审计](evidence/copmrec-unified-dualview-50k-20261005/inference-val-candidate42.json)与[固定100点轨迹分析](evidence/copmrec-unified-dualview-50k-20261005/training-trajectory-analysis.json)。
- [Validation独立复核](evidence/copmrec-unified-dualview-50k-20261005/validation-independent-review.json)与[有限bad case独立算术核验](evidence/copmrec-unified-dualview-50k-20261005/bounded-badcase-analysis.json)。
- [实际累计账本](evidence/copmrec-unified-dualview-50k-20261005/cumulative-budget-latest.json)与[v5.1完整研究报告](copmrec-unified-dualview-50k-research.md)。
- [实施凭据字段修正](evidence/copmrec-unified-dualview-50k-20261005/implementation-schema-fix-verification.json)：首次本地审计KeyError及原冻结字节保留，严格映射真实schema字段，不更改runtime或模型结果。


## 2026-10-06后续登记：closure保留，最后槽已分配但未启动

上文“最后槽未分配／仅只读鉴别”保留为v5.1结题时快照。其后一次既有输出只读核查已完成并独立复核：native42／v5／v5.1的cold总槽为216／6644／11113，分别包含0／6／8个正确cold命中；其余false cold为216／6638／11105。v5相对native的cold presence与新增／丢失case关联较弱，两固定组方向不同；v5.1的489个seen-target丢失中309个无cold。该事实不能证明冷占位造成损失，也不能推断Top11可恢复数量。[冻结只读记录](evidence/copmrec-unified-dualview-50k-20261005/cold-occupancy-existing-output-analysis.json)

现行训练content CE将cold logits替换为−100，而mixture仍使用未mask的完整logits，部署允许cold；这是baseline共用的既有支持集区别，不称bug或根因。结合保留v5的NDCG／Top5收益和v5.1反面结果，已将原150k内最后1train／50000及最后1完整Val固定分配给v5派生v5.2 `UnifiedFullCatalogCECoPMRec`，唯一干预是训练content CE包含完整目录cold logits。训练目标仍须seen，单query／单目录、三loss、learned alpha、历史资格及50k连续随机初始化协议保持；不恢复双目录平均，不扫描参数。

[实际stage registration](evidence/copmrec-unified-full-catalog-ce-50k-20261006/stage-registration.json)仅登记reserved，不代表启动：实际仍2train／100000、2完整Val、attempt3／失败1、新Test0，started2、committed100000。原3train／150000、3完整Val、3新Test上限不重置，未分配训练槽0，待启动保留槽1；本次新Testing与seed43成对预算均0。原双8%／成对复现目标保持active。本次实施、真实argv／Hydra、402文件/source16142f5a…3b9e0绑定、官方同步及实际CPU随机起点预检已完成；CPU forward0，训练／推理auditor54／86纯检查通过。双卡smoke已exit0，实际两local rank／1step／W&B run0验证通过且source一致，正式未启动；此处没有v5.2效果结果。完整主／辅预测及正／负／不确定结果决定见[固定v5.2方案](copmrec-unified-full-catalog-ce-50k-research.md)。
