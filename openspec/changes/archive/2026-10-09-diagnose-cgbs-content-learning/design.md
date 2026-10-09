## Context

原C128 checkpoint 07jcm4gj，content loss约8.2；宽MLP及E未改善C。此诊断不提出新机制，不以下降慢直接断言bug或梯度冲突。

## Goals / Non-Goals

目标：内容全目录排名/间隔、原始与0.1加权梯度强度/夹角、小批内容CE可拟合性。
非目标：真实推荐增益、test调参、自动完整训练、温度/权重搜索。

## Decisions

- 复用src.main inference与Trainer.predict、checkpoint恢复和AuxiliaryTensorWriter，不另建离线模型runner。
- training/evaluation明确分离，testing拒绝；按seed+user键hash选择，用户最终training窗口作为小批诊断样本，非原随机训练窗口分布。
- 默认训练16用户一批、两种100步临时拟合；evaluation128用户8批仅评分与梯度。单进程、FP32、inference_mode=false、eval关闭dropout，enable_grad计算梯度。
- 梯度按query、encoder分别计算generation和content，导出未加权norm、cosine及有效加权范数比；零范数夹角未定义。
- 小批拟合只在training，query-only与encoder+query各自从原权重出发，新Adam(lr=5e-4,weight_decay=1e-6)只最小化content CE。保存100步逐样本曲线，不保留权重；异常也恢复权重、requires_grad和RNG。
- 记录用户、输入hash、checkpoint张量hash、catalog身份、初始逐item logits、梯度批次键和拟合曲线。梯度为批次级，复制到用户行仅为writer形状契约，统计必须按batch去重。
- 拟合失败仅表示此预算/优化设置未拟合，不自动推出实现bug或不可表达。原始与加权梯度必须区分，局部负夹角不是全程因果证据。

## Risks / Trade-offs

- 小样本选择与真实训练窗口不一致 → 明确报告末窗口与样本hash，不估计总体指标。
- eval梯度与训练dropout状态不同 → 标记为确定性诊断。
- 全目录打分成本 → 最多128用户、batch16，拟合样本最多32，步数最多200。
- 辅助检查会暂时优化模型 → finally恢复并核验state/RNG，不配置checkpoint writer。
