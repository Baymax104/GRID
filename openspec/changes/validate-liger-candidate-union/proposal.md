## Why

偏好分散诊断已确认 mass 与 max 能找到互补目标路径，但单臂替换和深度条件路由均未把该互补性转化为推荐收益。当前唯一仍受证据支持的最小问题，是在不学习路由或排序器的前提下，保留两类生成候选后由既有内容分数统一排序，判断互补覆盖是否具有超过单纯扩大 beam 的价值。

## What Changes

- 增加两个固定推理臂：`mass_max_union20` 取 mass-beam20 与 max-beam20 的逐用户去重并集；`mass30` 仅把 mass beam 扩到30作为候选预算对照。
- 两臂均沿用联合 checkpoint、learned alpha、合法支持、cold-item union 与既有内容分数排序，不新增训练或可调权重。
- 为 union 臂保存 mass/max 源候选及并集候选，强校验并集精确性和目标覆盖 OR；保存标准候选 trace 以计算最终 Recall/NDCG。
- 增加三臂完成态比较：union 必须同时优于冻结 mass20 和 mass30，才能保留为探索性互补性候选。
- 冻结0次训练、2个正式输出臂和约3次 beam search；不调 beam、alpha、融合权重、quota、seed 或 subgroup。

## Capabilities

### New Capabilities

- `liger-candidate-union`: 在固定联合模型推理中生成、审计并比较 mass/max 候选并集与扩 beam 对照。

### Modified Capabilities

无。

## Impact

影响 LIGER 候选生成的可参数化入口、联合模型推理子类、辅助 trace writer、完成态比较、Hydra 配置、人工 launcher、聚焦测试和研究协议。默认 LIGER 与既有 checkpoint 行为保持兼容，不新增依赖；完整 prediction 仍由用户手动启动。
