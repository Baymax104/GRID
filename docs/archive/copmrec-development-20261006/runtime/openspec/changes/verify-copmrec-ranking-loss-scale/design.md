## 问题与预测

上一轮 6 个单 rank batch 中，4 个排序梯度范数为基础三项目标合计的 13–24 倍，共享主干方向冲突。若等权是重要原因，training-only 校准后的较小权重应提高合成梯度与基础梯度的对齐，并在固定短程续训中改善基础 CE、dense 检索或自然候选覆盖。

这是现有候选 CE 的优化强度鉴别，不提出新排序模块；与 prior art 的比较边界仍沿用 v1 设计。正例注入、候选生成、head 分数、基础三项目标和全部参数可训练保持一致。

## 参数与恢复

`ranking_loss_weight` 为有限正数，默认 1。总目标为原基础目标加该权重乘 raw ranking loss，日志继续保留 raw ranking loss。checkpoint 的既有 `copmrec_relevance.loss_weight` 记录实际权重。默认严格拒绝权重变化；`allow_ranking_loss_reweighting=true` 仅允许权重变化，并记录原权重、目标权重及来源 global_step。版本、候选分块、排序样本数、目录和 evaluation history 仍严格一致。

## 固定验证边界

- 来源为 f91njtjx 当前 validation-selected 本地 checkpoint，先独立复制并哈希固定，不使用不断变化的 last。
- GPU 1 单卡、batch128、累积2，每微批排序4例；两臂使用相同数据与 seed。与父 run 的双卡分片执行不同，因此只解释两臂的匹配差异，不宣称复现父 run 的精确轨迹。
- 固定 32 对真实 training 微批梯度：前16对校准，后16对保留检查。每对求两微批平均，校准权重 `min(1, 0.5 / median(rank_norm/base_norm))`，不根据 validation 指标选权重。
- 两臂分别权重1与校准值，共2000个 optimizer 更新，每臂1000；不扩预算、不追加第三臂。恢复同一模型/AdamW/scheduler 状态，scheduler 总长保持50000。
- 通过现有 `src.main experiment=copmrec_v1_1_train` 运行，独立 output_dir 和诊断 notes。固定终点 checkpoint用于匹配诊断，不以 best 选点冒充最终效果。
- 用同一2048个 outcome-independent selection 用户，比较 dense、相同候选内内容排序、hybrid，以及目标自然覆盖、content/SID CE；audit/test 不参与。
- 校准权重改善梯度且基础检索方向恢复：支持重新加权，完整从零训练收益仍待验证。仅梯度改善或稀疏指标无法判定：只确认优化尺度问题，保留收益不确定。无恢复/负向：收缩“等权是主要原因”的主张，不自动追加调参。

## 风险

中间 checkpoint 与 AdamW 动量已经受原训练影响，短程续训不能代替从零完整对照。样本指标较稀疏，要报告配对变化和区间；零更新梯度方向不能单独保证 AdamW 或推荐收益。
