# CoPMRec v5.2 阶段决定：保留部分收益，结束 cold 占位鉴别

## 已完成结果

唯一训练 [rha8mrvs](https://wandb.ai/baymaxam/GRID/runs/rha8mrvs) 已正常退出，实际连续完成50000更新、global batch256，即12800000次扩展后样本呈现。推荐模型从随机初始化开始，全部模块共同训练；没有外部推荐checkpoint、teacher、optimizer阶段重启或checkpoint pool。实际训练预算与同seed LIGER dense一致；本阶段三个模型共150000更新，不代表双方超参数搜索成本相同。

自己的100点训练期raw dense Validation选中47000步checkpoint。其真实参数、154份AdamW状态和50k horizon scheduler已核，best与last的实际保存状态均到47000；没有50000步终态完整状态文件。训练实际完成50000与保存状态47000分别报告。[训练审计](evidence/copmrec-unified-full-catalog-ce-50k-20261006/training-candidate42.json)、[独立终态复核](evidence/copmrec-unified-full-catalog-ce-50k-20261006/training-independent-review.json)。

完整单卡Validation [d7lcftto](https://wandb.ai/baymaxam/GRID/runs/d7lcftto) 已exit0，物理GPU7映射本地[0]，单进程。175个原始文件、22363用户、标签、keys、历史资格、合法唯一输出、实际checkpoint和402文件source均由[原始输出审计](evidence/copmrec-unified-full-catalog-ce-50k-20261006/inference-val-candidate42.json)核验。以下采用同history资格的完整部署评价，训练期曲线不能替代这些数值。

| 方法 | R@10 | NDCG@10 | 相对native的R增益 | 相对native的N增益 | 决定 |
|---|---:|---:|---:|---:|---|
| LIGER dense native42 | 0.0969458481 | 0.0540488986 | — | — | 固定对照 |
| v5 | 0.1004337522 | 0.0601508468 | +3.597786% | +11.289681% | 保留此前部分正向证据 |
| v5.1 | 0.0953807629 | 0.0558456247 | −1.614391% | +3.324260% | 固定双目录平均已否定 |
| v5.2 | 0.1014622367 | 0.0610582814 | +4.658672% | +12.968595% | 当前保留部分正向模型 |

v5.2对native的paired绝对差CI95：R为[0.0007590663,0.0082278764]，N为[0.0048093198,0.0093815752]，两个下界均正。相对冻结v5，R仅+1.024043%，CI跨0；N+1.508598%，CI[0.0000022124,0.0018431269]下界极接近0。本次固定2000次bootstrap所得单点区间仅提供较弱的N增量证据，不能称完整目录CE带来稳定、全面或至少3%的独立增益。

主门禁保持相对native两个指标各至少8%、两个paired绝对CI下界为正。实际N通过，R未通过，整体门禁false。当前2269个Top10命中，达到R门槛至少需2342个，即仍缺73个净命中；这只是整数门槛换算，不是已有数据证明可以恢复的案例。[固定门禁](evidence/copmrec-unified-full-catalog-ce-50k-20261006/candidate42-validation-gate.json)。

## 五项路线判断

1. **核心可反驳假设。** 共享seen残差的单模型能在一次连续50k中取得相对同预算LIGER的推荐收益；本次具体干预进一步检验完整目录content CE的直接竞争监督能否减少cold占位并改善覆盖。整体效果、干预增量和机制支持分别判断。
2. **支持证据。** v5的NDCG／Top5增益已经核验；v5.2相对native两个指标均有正paired CI，达到用户允许保留的明确部分收益。完整目录CE大幅减少cold错误占位，辅助点方向得到支持。对v5的Recall增量仍不确定，不能把整体收益全归于该干预。
3. **反证、门槛与缺口。** v5.1固定0.5／0.5共享query双目录平均相对v5两指标均下降且CI全负，具体机制已停止。v5和v5.2均未达到原双8门槛；未通过门槛不等于整体残差机制无效。seed43配对复现、新Testing和跨数据集效果尚未运行，不能声称可复现双8或泛化。
4. **关键实现疑点与未决项。** 实际配置、梯度支持集、源码归档、checkpoint及原始输出审计通过，没有阻断本次取舍的关键实现疑点。cold占位下降没有解释全部seen目标损失；剩余用户条件排序原因未知，允许保留该不确定性，不追加穷尽性排查、Top11推断或超参数扫描。
5. **累计预算与取舍。** 原阶段3train／150000更新、3完整Validation均已实际用完；Validation实际启动attempt4，其中一次预测前失败仍保留，新Test0。预算不重置，无未分配训练或Validation槽。保留v5.2部分正向结果，结束本次cold竞争支持集鉴别，不自动追加模型或评价；整体目标保持active、未完成。[实际累计账本](evidence/copmrec-unified-full-catalog-ce-50k-20261006/cumulative-budget-latest.json)。

## Bad case与适用边界

下表来自一次只读[已保存输出检查](evidence/copmrec-unified-full-catalog-ce-50k-20261006/bounded-cold-result-review.json)，没有产生新分数、排序、训练或评价。

| 量 | v5 | v5.2 |
|---|---:|---:|
| false cold Top10槽 | 6638 | 44 |
| 有cold槽的用户 | 3657 | 42 |
| 已命中seen目标前的cold槽 | 209 | 0 |
| 上一行的seen命中用户分母 | 2240 | 2269 |
| 相同2032个共同seen命中用户中，目标前cold槽 | 175 | 0 |
| 51个cold-target用户中的Top10命中 | 6 | 0 |

对v5新增237、丢失214、净增23；其中seen目标新增237、丢失208、净增29，cold目标损失6。两个预固定key组的全体净增分别21和2，seen净增23和6，改善幅度不均。移除cold槽伴随cold真命中损失；不能把这项干预表述为冷启动收益。

v5.2的221361个错误Top10槽中仅44个是cold。相对native的824个丢失用户中，只有1人的v5.2列表含cold；相对v5的214个丢失用户中只有2人含cold。继续压cold占位已缺乏主要瓶颈依据。对native Top5净增181、6–10位置净减80；824个丢失来自原Top5的444例和原6–10的380例。后续更有价值的问题是seen商品之间的用户条件覆盖与排序取舍，而非反复调整cold规则。

若保持当前全部分数和资格规则，仅移除cold商品，最多影响当前42个含cold的用户列表；即便这42人全部新增命中，也不足补足73个净命中缺口。这是受影响用户数的上界，不是Top11恢复预测，也不适用于会改变所有用户分数的重新训练。

这些统计只描述已有列表；没有目标的完整候选分数或Top11，不能判定哪73例可恢复，不能由cold频次归因query或目录，更不能把共同命中位置改善当作全部Recall损失的解释。

## 证据限制与下一动作

Beauty单seed Validation用于开发选择，固定checkpoint条件下的逐点CI没有为整个搜索历史提供多重比较保证。native42历史训练source缺口不由本次新402文件归档回填。参数数目或更新数相等不证明FLOPs、GPU时长或搜索成本相等。

本阶段不新开Testing或seed43实验。下一固定方案及新增成本必须先形成可审阅记录并明确分配，保留原阶段成本；尚未排除某种原因不能作为增加预算的理由。当前保留的v5.2是正向基础，原双8、配对复现及Testing要求均不降低。
