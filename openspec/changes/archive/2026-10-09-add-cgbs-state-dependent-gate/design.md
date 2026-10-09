## Context

依据相邻研究记录 `validate-cgbs-content-routing-mechanism/gate-decision.md`：原 C 仍低于 A，exact 未改善原型，正确内容优于置乱。状态门控为待验证机制，不预设有益。

## Goals / Non-Goals

目标：实现独立 E arm，默认四层新增12参数，以相同初始化、训练预算检验状态适应性；提供可审计推理干预和必要统计。

范围外：自动完整实验、原型扩容、新损失、辅助梯度隔离、校准训练过程及自动发布。正式常量估计要在E有改进后按统一入口另行执行，本轮提供显式值/来源接口而不伪造已校准常量。

## Decisions

1. **领域 helper**：在 CGBS 目录添加纯函数计算 `z=[H(P)/log K,H(Q)/log K,JS(P,Q)/log 2]`。以detach后的float32 log概率计算，非法位置置安全值，单分支熵为0，0概率不形成0乘负无穷。特征截断到[0,1]以吸收数值误差。
2. **模型**：E=`content_init_state_gate`，保留原 `mixture_logits`，新增 `gate_weights[L,3]` 全零；`alpha=alpha_maximum*sigmoid(b_l+z@w_l)`，每个父前缀得到一个[N,1]权重。训练teacher和真实beam统一走conditional_log_probs；不增加decoder/query forward，不消耗新增RNG。
3. **梯度**：z停止梯度；P/Q原混合和辅助CE路径均保留。门控零初始化时旧参数梯度与C相等。每层行参与图，DDP不引入未使用参数，即使某层特征全零也保留有定义的零梯度。
4. **推理**：`gate_mode=dynamic|base|fixed`。base使用学习后的b，仅为敏感性对照；fixed要求每层显式有限值 `0<alpha<alpha_maximum` 和非空 `gate_constant_source`，记入metadata。未选fixed不得传固定值/来源，非E不得传干预，训练禁止base/fixed。E第一版只支持content_scoring=trained/off；exact/shuffled_exact/rerank在E明确拒绝，避免无状态rerank套用错误门控或把变化的门控误当固定alpha参考；旧arm所有模式保留。
5. **身份**：E增加版本化 `state_gate` checkpoint契约，包含固定的特征顺序、归一化与梯度规则；旧arm契约不增字段。推理gate_mode/常量来源不是训练身份，E checkpoint可在三种gate_mode下恢复，干预metadata区分。
6. **统计**：训练在原teacher调用中收集detach的每层alpha向量[L,N]，临时变量随batch释放，不挂模块状态、不额外forward。E专用model config通过现有MetricEngine重复定义Mean/Min/Max；输入为每状态数值，跨rank由torchmetrics按样本数/极值聚合。样本包括所有合法teacher父前缀（含单子分支），不把多层均值当作各层变化性证据。旧model config及metrics不变。
7. **入口**：新增薄gate model config（训练/推理），原root脚本按E选择该model group，用户override最后覆盖。mechanism `--condition e`只训练一次，默认both仍B/C；E screen仍dynamic trained/off。常量使用root inference及显式override，不增加自动矩阵。
8. **常量估计预定协议**：后续选择E best checkpoint；只从training文件的稳定字典序、单worker有限一轮读取，按确定性SID/完整历史预处理生成每条记录最后一个next-item样本（不做随机causal expansion）。训练输入不保证保留user_id，因此以相对训练文件路径和文件内零基记录序号作为样本身份，取SHA256(`seed42|relative_file|record_index`)顺序最小的最多4096条有效记录；不宣称这些记录对应唯一用户。teacher正确父前缀上按样本平均每层alpha。记录输入文件hash、记录清单/hash、输入artifact、checkpoint ID、实际样本数及每层和/计数。不得混入evaluation/test。该均值仅为匹配参考，不是最优常量；本轮不执行或宣称已提供该校准runner。

## Risks / Trade-offs

- 特征无法识别误伤、teacher/beam分布差异 → 维持单条件、先看A/C最终推荐，不以loss或门控变化证明机制。
- 小概率/单分支数值或sigmoid饱和 → 极端logits及正反向测试；动态alpha安全夹到float32可表示的开区间，避免log(0)；不增加新的可调超参。
- telemetry同步开销 → 只取已有teacheralpha，不记录全量状态或重跑模型；不报告未经测量的速度提升。
- 常量值来源声明不等于自动数据验证 → metadata保留来源，科研使用前核验校准产物，不把手工列表视作已验证均值。

## Migration Plan

独立arm与model config，不迁移历史checkpoint。保留原C、D脚本；聚焦测试/compose/shell/OpenSpec验证后Mutagen flush并确认三个session。E完整训练由用户手动执行，GPU0/1。科学结果仍待运行。
