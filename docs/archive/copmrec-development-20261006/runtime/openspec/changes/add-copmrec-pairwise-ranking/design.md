## Context
beta网格0/.1/.25/.5/1的selection选择.5但audit净损14，触发条件排序学习。旧失败记录保留，当前只实现一个固定实例。
## Goals / Non-Goals
目标：训练侧监督减少错误替换；非目标：主模型微调、候选扩容、evaluation标签拟合、架构/损失扫描。
## Decisions
缓存使用training每用户最后一个完整商品作为目标、之前历史为输入，保持mass20+cold，仅一次搜索。缓存默认本地，manifest记录payload/来源checkpoint哈希与完整协议。

排序器输入五项候选内特征：内容/混合/生成标准化分数、混合与内容标准化差、内容分数到Top10截断分数的标准化距离。5→8→1 ReLU MLP，最后层零初始化，初始严格等于内容排序。输出tanh残差，乘内容标准差后加原内容分数；没有额外ID、文本网络或主模型参数。

仅training已覆盖目标参与pairwise softplus，使用内容排序最高的最多10个非目标竞争者；未覆盖用户仍记录并保留evaluation全体统计。固定残差平方惩罚0.01、Adam lr0.001、batch256、10epoch，无early-stop/调参矩阵。validation仅用固定用户hash selection半集选择best val/ndcg@10；最后推理使用audit半集。二者仍属于已看过的开发evaluation，不称独立确认。

只训练小型head，主模型不进入训练实例。三个完整运行槽位：training缓存、head训练、audit预测，均用户手动启动。
## Risks / Trade-offs
主checkpoint见过training标签，候选与分数质量可能比evaluation乐观；需看开发复验净收益。小型MLP仍可能过拟合，不以训练loss下降作效果证据。
## Migration Plan
旧copmrec_ranking_v1保持严格校验；新增copmrec_learned_ranking_v1保存learned第四臂及旧三臂分数。数据cache来源签名进入head checkpoint，推理拒绝不匹配cache。
## Open Questions
真实训练缓存尚待手动生成；没有新的排序学习效果结论。
