## Context

原A使用token_content_init/full_content、合法条件softmax和四层完整SID，CE为路径NLL的层均值。需要判断A自身的精确排名与实际beam差异，不能加载旧MIR审计替代。

## Goals / Non-Goals

目标：默认Beauty seed42 checkpoint 5g3wpbg7、evaluation、128个hash用户、beam10、单进程，发布可校验的逐用户审计。
非目标：本轮不实施G1排序训练、不重新训练、不自动同步/运行、不扩大beam或访问testing。

## Decisions

- 新审计子类继承TigerCatalogGrounded，不增加参数/buffer，不改变checkpoint contract；只允许原A，无二层分支或推理干预。加载时沿用原契约并计算state_dict SHA256。推理前必须已加载且eval、单进程。
- 复用ItemResolutionAuditDataModule的hash采样与数据契约（其处理不依赖MIR模型），显式standard/evaluation、batch1/workers0。默认hash seed20260918；有限1..512用户和1..1024 chunk。
- 复用encoder一次，全部catalog SID分块teacher forcing，各自合法前缀log概率之和为item logp。检查有限与总质量归一化，不二次归一化掩盖错误。
- 调用原A的实际_beam并采集target prefix survival；真值只用于观测，精确评分和beam选择不依赖标签。验证beam返回项分数与精确teacher概率一致。
- 排名以概率降序、同分item key升序。额外保存atol=1e-5的目标rank下/上界，near-tie不能武断归因。beam未命中且上界<=K才记确定搜索遗漏；下界>K才记确定评分失败；其余为边界不确定。
- 共享AuxiliaryTensorWriter保存a_score_search_audit.pt，含完整概率、beam keys/logp、目标rank/NLL、前缀生存、计时、输入/目录/checkpoint身份。独立validator从原始概率重算rank、TopK和分类，保护数据有效性。
- 可视化/聚合后续直接读取这个证据，128用户的稀疏命中不作为显著性依据。

## Risks / Trade-offs

- 枚举约155万路径有真实成本 → 仅固定128用户、encoder复用、分块，计时分开记录，不承诺GPU耗时。
- 不同batch形状产生微小数值差 → 保存rank边界，检查beam评分一致，不将数值tie说成搜索错误。
- 当前目录概率TopK不等于相关性上界 → 同时记录beam与精确命中，允许beam偶然命中精确TopK以外目标。
- 用户覆盖脚本默认参数 → 保留覆盖能力并记录resolved config；模型仍强制A/evaluation，不以脚本默认替代校验。

## G1 固定候选协议

仅在G0支持评分优化时另建实施变更：原A19000步最佳权重，两个分支均新建Adam lr5e-5、各2000更新、每500步验证；CE对照与CE+lambda0.1/K2完整SID采样排序，训练时uniform重采样合法负item并排除当前目标及已观察历史。保留按层平均CE，排序使用完整logp和。不使用testing或二层交互checkpoint。主判据比纯CE和冻结A均+1% best-val NDCG、同点Recall不降、末三点平均不低于CE；记录相同步数和共有wall-time指标，吞吐降>20%先调查。不声称新算法。
