## Context

v0已有inference_mixture_alpha，但训练时明确拒绝该字段。现有JointConstantGate只有一个可学习bias，alpha=sigmoid(bias)，初始化不消耗RNG；其混合NLL与合法Mass候选共用bias。

## Decisions

- 新增fixed_mixture_alpha，默认null，当前固定配置0.8，支持有限开区间(0,1)。只用于v0，拒绝叠加content_only、机制对照或推理覆盖，避免改变其他版本协议。
- 保留dynamic_gate.bias状态键与模型初始化随机序列；固定模式设bias=log(alpha/(1-alpha))并requires_grad=false，混合NLL仍训练两路模型，训练日志报告固定alpha。
- 候选处理使用同一冻结gate，保证teacher forcing与逐步概率一致；trace标记fixed_training，另记录checkpoint值与固定策略。
- 保存copmrec_alpha_policy、copmrec_fixed_mixture_alpha。旧checkpoint缺少新字段按learned处理；固定与学习模式不得交叉恢复，不同固定值也不得混用；严格state加载校验冻结bias。
- 新配置继承v0，其数据、optimizer/scheduler、dense验证、content终排、trainer及源码快照保持；不提供旧checkpoint，随机初始化开始。薄根脚本复用liger_common参数解析，支持dry-run/notes/末尾override。

## Verification

CPU合成单测独立枚举0.8混合概率、检查teacher forcing一致、两路非零梯度及优化后gate不变，初始化除bias外逐项一致，checkpoint恢复/拒绝及预测trace一致；Hydra compose对比v0、脚本语法/quoting/错误输入/torchrun验证。远端实际v0配置与同步SHA核验后进行显式零学习率两进程1-step dry-run，仅验证装配、梯度和DDP，W&B关闭、不保存checkpoint、不作效果证据。

## Risks

固定权重改变整段训练轨迹，单seed效果不能推出全局最优alpha或固定训练普遍优越；val dense选点与候选评价边界保持。alpha用FP32 logit表达存在正常舍入，日志与trace明确策略值0.8；恢复校验避免固定bias被旧学习状态覆盖。
