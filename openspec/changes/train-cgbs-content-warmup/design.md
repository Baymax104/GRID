## Context

一次固定5k/35k、seed42、40k总更新预算，对照直接联合训练ld5ecf58及A dragtsrn。不将新增阶段描述为已解决过拟合或目标冲突。

## Goals / Non-Goals

实现可恢复的训练阶段切换，验证参数更新范围；不改变在线评分、目录或网络容量，不自动运行实验。

## Decisions

- 训练子类使用Lightning global_step：0..4999仅未加权内容CE；5000起原generation CE+0.1 content CE。共享Encoder及SID embedding、query更新，Decoder独有参数及mixture无梯度；不切换requires_grad、不重建Adam。
- DDP明确find_unused_parameters=True支持预训练阶段未使用Decoder；不用static_graph。
- 独立组件配置指定模型、DDP和checkpoint callback；复用根训练脚本的Hydra透传，无新增shell入口。
- 训练输出记录training_phase（0内容、1联合）；预训练generation_loss占位0但以阶段为MeanMetric权重，日志无有效generation loss（NaN），不把0当实际生成损失。
- 验证继续每500步运行，val/loss仍是teacher-forcing生成CE，不是验证内容CE。checkpoint callback跳过global_step<=5000的保存，只在联合阶段选择最佳NDCG。
- checkpoint记录训练schedule，训练子类恢复时必须一致，global_step及optimizer由Lightning恢复。推理可使用原CGBS类加载相同state_dict与catalog_contract。

## Risks / Trade-offs

共享embedding会改变Decoder的输入表，但不更新Decoder专属权重。5k预训练使Decoder只有35k更新，因此等总step不等FLOPs或等Decoder更新数；报告实际耗时，不宣称纯初始化因果结论。预训练不消费Decoder dropout，联合阶段随机流不会逐值匹配直接训练。此为训练顺序方案的整体对照。

同一run可在阶段边界前后恢复；禁止用旧联合checkpoint假装从头预训练。dry-run仅做有限预训练烟测，不能验证5k切换；CPU回归专门验证边界。
