## Context
本轮归一化扩展：`pairwise_normalization=training_mean_cap`仅作用于competitive NDCG。从已核验training bundle的covered样本原mixed分数计算平均pair权重和C，CPU FP32一次计算、冻结，不读取evaluation来拟合。主loss分母为min(Z_u,C)，epsilon安全不变。默认per_user沿用旧目标。C、covered用户数和版本写入目标契约，缓存fingerprint继续沿用checkpoint cache契约；fit/validation初始化与audit开始重算核验，旧NLL禁用新归一化。W&B记录配置模式、运行时目标契约及分母/增强比例日志。小批量探针已支持实施，新固定预算为一个训练加一个audit，用户手动启动，匹配klqyxo8m/pvhjwp9q的2160更新，旧阶段不重置。
training i05boy01覆盖3454用户，evaluation hpfflo20固定哈希selection/audit。保留旧实例3/3负结果，不复用旧剩余预算。
## Goals / Non-Goals
目标：训练混合分数本身，匹配原NLL对照。非目标：改baseline、搜索新候选、调整alpha、自动完整运行。
## Decisions
原checkpoint初始化，两臂只训练decoder非共享参数，冻结共享embedding/绑定输出头、encoder及内容投影，冻结模块eval。缓存目标对应原training/evaluation最后商品，历史来源签名与checkpoint契约一致。
排序臂用当前混合排名交换的DCG10变化权重乘softplus负正分差，NLL臂用目标混合路径平均负log概率。两臂均保留SID CE，权重1。AdamW lr1e-5/weight_decay.01，batch8，accumulate4，20epoch，每2epoch选择val/ndcg10最优，梯度clip1，FP32；无参数扫描。每epoch432个microbatch/108更新，共2160更新。
初始checkpoint和候选固定；评分decoder更新不参与retriever搜索。统一audit一个入口载入两臂best，输出ndcg臂并保存content/mixed/generation/nll/ndcg五路trace；四种比较含ndcg相对nll配对CI。
## Risks / Trade-offs

## 双分支正确关系保护

用户采纳bad case提出的双分支参照。preservation_teacher_scope默认content，显式content_generation启用原缓存generation teacher。两teacher分别在training真值目标进Top10、严格排在负例前面时给出DCG10折扣差；同pair参考间隔取最大，重复pair不相加。竞争集合仍content/current mixed各Top20并集，分子分母仍相同mask。weight1、SID CE1、原初始化/20epoch预算、冻结边界不变；不在推理按真值切分支。内容单teacher与关闭目标保持旧checkpoint契约，双teacher版本改为correct-content-generation-discount-margin-v1并严格拒绝误加载。

最小training探针检查新增teacher正确关系有实际梯度、teacher detach、重复pair无翻倍、对未支持样本不增加梯度，以及新增项对content参数梯度的关系。若明显破坏content正确方向，先收缩而不发起完整训练。若通过交付固定训练+audit共2个用户手动run，比较7q2njodh/nwxzx1qs的单content保护，不重置旧预算、不扫描teacher/weight。主门槛及负向/不确定停止决定保持。

## 2026-10-02内容正确关系保护

用户采纳competitive失败样本后的建议。仅在显式content_preservation_weight>0的competitive NDCG臂启用，默认0兼容既有臂。teacher是冻结缓存content排名：d(r)=1/log2(r+1)（r<=10），否则0。目标原content进Top10时，在content/current mixed Top20并集负例中，保留content严格低于目标的商品；分数相等不提供保护。teacher参考间隔m=d(r_y)-d(r_j)>0，权重也是m。每用户损失为sum(m*0.5*relu(m-(s_y-s_j))^2)/sum(m)，最后对完整batch取均值，未命中/无pair用户贡献可微0。teacher全部detach，不将原始content logit间隔套给混合路径分数；discount以固定单位温度映射为teacher logit，是显式设计选择而非已校准概率。达到参考间隔后零梯度，不强行拉回更好的mixed关系。

新增固定weight=1实例，与competitive NDCG+SID CE组成总目标。不自动调整权重或搜索温度；training小批量若不能提供有意义的保护或明显破坏恢复方向，则停止交付完整训练。该参照项与标签监督存在重合，必须通过匹配最终收益鉴别，梯度探针不证明泛化。

启用时checkpoint目标契约新增保护版本/weight；禁用时保持旧契约字典不变。NLL只允许weight=0，双臂audit创建NLL时强制0、NDCG保留配置，避免将NDCG目标污染NLL。推理分数不依赖新增保护项或目标标签。

用户采纳后新阶段上限2个完整run（固定修改臂训练+一次audit），旧阶段预算/否证保留，完整运行用户手动开始。保持原step48000初始化、20epoch/2160更新、best选择及同候选，使用29o9dx80/zy940m5u为匹配competitive对照。主要门槛不变，负向/区间跨零结束该实例，不扩成权重扫描。

## 2026-10-02有效竞争pair
用户采纳已有bad case分析并授权实施。保留原全部候选的混合评分及推理，仅在NDCG训练损失内提供competitive scope：原content和当前mixed各Top20候选的并集，去掉padding与正目标，排名/集合detach，同mask作用于分子和分母。默认all保持旧目标；只有显式override启用competitive。NLL忽略pair scope，保持原目标。记录有效负例数和保留权重比例。

新增checkpoint objective_contract记录NDCG scope/k，加载时严格校验；旧checkpoint无此字段按all/k20解释。audit共享模型配置可以启用新NDCG目标，NLL契约仍为空且不改变评分；trace记录两臂目标契约。对旧NDCG臂的匹配比较复用ri2813bi原trace按用户键连接，不能把NLL当作归一化的唯一匹配对照。

新阶段最多2个完整run：一个competitive训练，一个统一audit；旧阶段4/4剩余0保持。初始化仍原step48000，超参数全部匹配旧NDCG，旧NLL和旧NDCG均不重训。只授权实现/交付，完整run仍用户手动开始；新stage不扫描k/权重/epoch。若相对content未满足原门槛，或仅通过阻止全部变化回到content，结束该具体实例。

训练需要重新可微解码53候选，成本高于小head，chunk4限前向规模。固定池成功不能证明新decoder的beam收益。验证每轮冻结参数哈希、数据来源与实际内容分数，拒绝契约不匹配。

## 2026-10-02效率修复
用户报告训练慢及GPU低利用率。NLL损失只依赖已覆盖目标，允许仅解码该目标，先检查原池覆盖以避免补目标；排序臂及验证仍解码全池。两臂损失定义、batch8、累积4、FP32、学习率及20epoch不变。候选chunk由4增至64，减少小kernel/前向调度；decoder dropout随机数分配随形状变化，因此不承诺训练轨迹逐字节相同。eval下核验chunk间分数及梯度等价。冻结目录投影缓存只在内存持有，不进入checkpoint或发布；设备变化时重建。cpu_threads=4在model config记录并设置PyTorch intra-op线程，替代node1默认96。完整实验仍用户手动开始；原xdcm511f中断记录保留，续训时须单独记录实际run数，不能把物理重启隐去或重置旧阶段预算。

## Top10边界辅助目标
排除正例后的第10个有效负候选以detach稳定排序确定，双方原分数保留梯度；softplus(s_negative-s_positive)，不足10个负例的用户贡献可微零，全batch取均值。默认boundary_gradient_fraction=0，启用值0.25只允许competitive NDCG+training_mean_cap。lambda以原training covered混合缓存计算：fraction乘主目标正例分数梯度均值除边界正例分数梯度均值；CPU FP32、冻结、记录版本/K/fraction/lambda/covered和eligible人数，缓存契约沿用现有来源。仅training可拟合；fit/validation/audit重验，checkpoint不兼容目标拒绝；旧NLL强制关闭。保留现有分数公式及NDCG/双teacher/SID，日志记录边界loss、eligible、gap和lambda。32用户零更新探针是完整实现前的鉴别；不能从独立分数梯度推断推荐增益。
