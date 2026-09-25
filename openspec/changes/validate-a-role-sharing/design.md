## Context

A通过同一SID表进行encoder历史查表和decoder前缀查表，输出头独立。原审计可复用合法目录精确评分、checkpoint指纹与通用writer。

## Goals / Non-Goals

目标：验证角色共享的局部损伤信号；不训练、不认定全局瓶颈或新颖性。

## Decisions

- 使用统一predict入口及单进程FP32，inference_mode=false，显式enable_grad仅作用于临时embedding张量。
- datamodule选择确定性training末目标32用户及与之互斥的evaluation64用户，将支持批次显式暴露给审计模型；首次predict先计算支持方向，再处理评价样本。全目录遍历仅用于hash抽样，不做模型全量计算。无随机窗口增强，结果不等于训练全过程梯度。
- 前向hook按阶段替代共享查表输出；模型参数全冻结，hook在finally中移除。不修改现有训练类或checkpoint结构。
- 每个support用户核验零干预CE等价及g_shared=g_history+g_prefix。方向使用support均值，evaluation标签绝不参与构造。
- 条件为baseline及shared/split/history/prefix/antisymmetric/random_antisymmetric，扰动占两张角色表联合范数的0.001/0.003。所有条件按联合范数匹配；随机反对称方向逐行匹配确定性反对称方向幅度，以控制活跃token分布。
- 原始模型不变；每个条件重新编码、全目录精确评分。导出所有条件的目标CE、精确rank与TopK，不称GPU速度提升。
- 保存support每用户每token的角色梯度范数/内积/曝光次数、训练方向指纹、样本输入指纹和checkpoint/catalog身份。仅记录诊断结论待分析，不自动判定瓶颈。

## Risks / Trade-offs

- 小幅扰动只验证局部可塑性，不能替代匹配短续训。双幅度不得事后择优；随机控制只是一条预定方向，阳性后必须复核。
- best checkpoint不能代表初期；稀疏末目标与实际窗口采样不同，明确限定范围。
- 精确排名增加时间；默认64 evaluation用户，可dry-run预检，完整运行由用户开始。
- encoder角色包含现有输入流中的masked target占位；统计的是实际有效查表曝光，不把它误称纯历史item频率。原有数据协议保持不变。
- dry-run通过统一launcher关闭结果发布并只评价1批，支持集缩为2用户，仅保留shared/split的首个幅度。它不能用于机制结论。
