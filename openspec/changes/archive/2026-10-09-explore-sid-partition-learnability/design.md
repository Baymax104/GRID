## Context

量化候选尚无因果资格。旧路线全部证据见research/docs/2026-09-18-route-termination-review.md；新颖性边界见research/docs/2026-09-18-sid-quantization-question-screen.md。A收益不证明量化分组有问题；静态damage、局部负迁移与早期beam解释均不能直接移植。

## Goals / Non-Goals

目标：实现一个可重放、training内无用户泄漏的CPU代理资格实验。非目标：训练新推荐器、修改SID、证明量化因果瓶颈、宣称论文新颖性、恢复暂停方案。

## Decisions

### 输入与抽样

- 一个Beauty RKMeans设置；显式training目录，raw SID bundle和内容embedding bundle。首层码是唯一分析分组，不扫描深度。
- 一用户一条最长训练序列，仅末两项构成(context item,target item)。同用户重复前缀取最长；非前缀兼容记录直接报错，不猜测时序。
- SHA256(seed,user)排序后最多50000用户，以序号模4分为fit-A/fit-B/eval-A/eval-B。两次重复使用完全互斥用户，不能当训练seed复现。
- 选定转移中的所有item必须在目录；SID为三列语义码或项目标准四列（含去重digit）非负整数，四列必须唯一，embedding严格按key对齐且有限非零。首阶段不做唯一SID命中评价，raw碰撞额外行数只记录，不冒充碰撞item占比。
- 正式读取上限1000000行；超出即失败，不静默截断。dry-run最多读取128行、选择64用户，缩小重采样，强制smoke_only，不产生正式判断。

### 固定统计

两次重复分别fit-A→eval-A与fit-B→eval-B。上下文固定为前一个item原始ID，避免同时改变输入分组。目标为首层码g。

频次基线q(g)=(n_g+1)/(N+G)。条件估计p(g|x)=(n_xg+20q(g))/(n_x+20)。上下文未见时严格回退q。单用户gain=log(p/q)，单位nats；只用fit计数。

每重复做99个固定seed置乱：仅打乱fit记录的context，保留context/target边际和eval不动。输出每次置乱均值、观测均值和单侧Monte Carlo p=(1+#null>=obs)/(100)，这是条件关联筛查，不是量化干预。

用户bootstrap1000次输出gain均值95%区间。分组仅保留两次eval均至少20用户且两次fit均至少20目标的组。每重复按组聚合gain，用OLS移除log1p(fit支持)、log1p(eval支持)、log1p(目录组大小)、组内单位embedding的均方离散度，保留截距；输出残差Spearman及组bootstrap区间（描述性、非因果，不包含全部估计误差）。

### 资格门槛

数据有效性失败抛错。少于30个合格组或每次覆盖率不足50%，判insufficient_support；不得为通过而降低阈值。两次gain区间下界>0且两次置乱p<=0.05，才通过context_signal；残差相关>=0.5且其区间下界>0，才通过group_reliability。全部通过输出proxy_qualified，不自动批准后续训练；其余有足够支持时输出no_go_for_this_proxy。正向门槛是探索决策线，不是会议录用或严格因果标准。

### 集成与输出

data helper负责输入校验、TFRecord读取、去重、哈希和目录装配；datamodule只提供一个CPU test batch。quantization组件负责纯统计和Lightning test_step。共享StructuredAnalysisWriter输出summary.json、protocol.json、identity.json、users.csv、groups.csv、null.csv；共享lineage callback记录解析的输入Artifact。默认W&B启用，允许logger=null和publish_wandb=false本地使用。dry-run通过统一入口关闭writer/logger。所有输出均在paths.output_dir内。

## Risks / Trade-offs

- 单末次转移代理弱于Transformer → 不把失败推广为所有量化无效；不通过即停止该代理而不是扩充上下文扫描。
- 支持不足 → unavailable，不把缺失组填零；明确被排除样本比例。
- OLS只控制选定混淆 → 不称独立因果证据，后续仍需受控分组干预。
- 内容离散度不是量化重建误差 → 字段使用semantic_dispersion，不混用MSE术语。
- 跨用户拆分仍有共享item → 用户CI是条件性诊断，不能替代跨训练seed和独立数据集确认。
- 只有首层统计 → 不声称后续路径或完整item排序已改善。

## 后续探索协议

E2仅在proxy_qualified后进入设计：原SID/受限行为重分配/匹配扰动三臂，各seed42最多20k更新；最多两个新映射、三个训练、三个开发预测、三个128用户精确评分。完整唯一SID成对交换保留tuple集合、长度、容量，匹配改动数和几何/频次差；构造约束失败即实验无效。不得直接用旧模型评估新SID。两类改动同时影响历史与目标，只解释整体编码效应。首轮共享随机初始化规则隔离内容初始化交互。以相对两个对照NDCG至少+1%、Recall不降、机制同向作为工程晋级线，仍报告配对区间；所有条件固定同20k和每500步validation、best-val选择。

E3仅E2通过：映射与配方冻结，seed43同三臂，每臂20k更新，最多三个预测及三个精确评分；不复现就停止当前机制。E2/E3合计六训练120k更新，独立运行计数明确，均未实现。Beauty旧testing不参与规则/门槛选择。后续论文还需强近邻、内容初始化、未用设置及效率确认，当前无运行授权。
