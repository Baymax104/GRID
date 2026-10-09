## Why

v4 的共享协同残差在验证集相对匹配续训对照产生正向增益，但当前终排仍只使用 content 分数，未使用同一 checkpoint 已监督学习的完整 SID 混合概率。本变更仅检验新 v4 表示下，固定的混合路径概率终排能否改善同候选推荐；候选内遗漏数量和旧归档实验不作为收益证据。

## What Changes

- 添加 `final_ranking_mode=content|mixed` 推理策略；content 为默认，mixed 不引入参数、训练或 alpha 扫描。
- 在同一次搜索产生的 `gen20 ∪ all_cold` 上复用合法条件概率评分，按完整 SID 的逐层混合 log probability 之和排序。
- 通过 Liger 默认 content 评分 hook 集成，不复制 retrieve，保持旧默认数值行为。
- 返回真实最终排序分数；mixed trace 记录同候选 content 参考排名、TopK 与完整候选分数，严格核验配对关系。
- 仅安排一次固定 checkpoint、完整 evaluation、单进程推理验证；配置、状态和正式运行由主代理统一管理。

## Capabilities

### New Capabilities

- `copmrec-v4-fixed-mixture-ranking`: 严格兼容 v4 checkpoint 的无训练混合终排与同候选 evaluation 配对证据。

### Modified Capabilities

无。旧默认 content 行为和训练 checkpoint 参数契约保持。

## Impact

涉及 Liger 最小候选评分/trace hook、v4 推理策略、已有 mixture ranking 评分核心、trace validator 和 CPU 测试。后续使用统一 `src.main` / Hydra 入口、共享 writer 与已有 Artifact lineage。训练过程不改变；旧 v0 排序实验和退役入口不重启。
