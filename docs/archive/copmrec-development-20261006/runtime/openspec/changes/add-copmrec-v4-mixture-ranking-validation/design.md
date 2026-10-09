## Context

当前 v4 在完整 evaluation 上的 best NDCG@10 为 0.04882855，匹配 scale0 续训对照为 0.04786393；六个等步数比较均为正向。该结果支持新残差表示有作用，但没有证明完整 SID 混合概率终排更优。当前 mixture NLL 已监督逐层合法混合概率，最终部署仍使用 content 排序。

现有 `mixture_ranking.py` 保留技术评分实现；旧 v0 experiment 已退役，本变更不恢复它。主代理已批准在新 v4 固定 checkpoint 上执行一次 evaluation 配对验证；实现代理仅修改相关代码和测试，不启动远端作业。

## Goals / Non-Goals

目标是新增 `final_ranking_mode=content|mixed`、在完全相同候选上比较两种终排，并保持参数、输入、checkpoint 契约和默认 content 数值兼容。

不新增训练、head、参数、alpha 设置、候选搜索或 checkpoint 扫描；不使用 Testing 选策略。853 个候选内漏排只说明潜在容量，旧归档 mixed 结果仅解释技术依赖，均不作为当前正向效果。

## Decisions

1. Liger 新增默认 content 的 `candidate_ranking_scores(encoded, mask, content_logits, user, rows)` hook；retrieve 继续负责一次搜索、合法候选 union、稳定排序与真实最终分数返回。新增小型 trace hook 仅在标签存在时补充诊断字段，评分接口不接收标签。
2. 将 `mixture_ranking.py` 的 batched teacher-forcing 算分核心抽取为复用函数；旧 class 委托该函数，v4 对每用户完整候选调用。decoder 输入为 BOS 与 SID 前 H-1 位，逐层通过实际 `candidate_processor` 获取条件混合 log probability，完整 item 分数为 H 层之和，不附加 EOS 或长度惩罚。
3. 同一 checkpoint 的 learned alpha、Mass 聚合、完整目录和 cold 支持集保持；不对候选子集归一化。alpha1 的完整分数 telescoping 为 content log-softmax，alpha0 为合法 conditional generation，eval 时负的目标分数均值除以 H 等于 mixture NLL。
4. `content` 默认不增加状态参数、不改变输出 tensor、搜索或 RNG；`mixed` 只允许推理/验证，fit 和 training=True 监督调用明确拒绝。最终排序配置不进入原残差参数契约，已有 v4 checkpoint 严格兼容。
5. mixed trace 使用明确的 `final_score=mixed_full_sid_log_probability` 和 v4 专属协议；补充 padded candidate rows/SID、content/mixed scores、同候选 `hybrid_content_rank` 和 `content_topk_sids`，协议记录完整 cold rows。validator 校验候选 union、padding、有限分数、stable 排序、覆盖关系、两套排名和输出对应关系，mixed 不套用 content 的 hybrid_rank<=dense_rank 限制。
6. 一次单进程 evaluation 输出同时记录 mixed 主输出和 content 参考，先复算两套指标与逐用户损益再决策。marginal_probs 返回实际 mixed 分数，CPU 检查其与输出次序及指标一致。主代理管理运行配置、同步和预算。

## Risks / Trade-offs

- 完整路径概率不保证 Top10 更优 → 不宣称收益，固定一次 evaluation；负向/不确定时保持 content，不自动追加扫描。
- 重排可能恢复漏排同时丢失旧命中 → 成对报告新命中、损失、共同命中位置及 Recall/NDCG 差值。
- 候选 decoder 扩展增加显存和耗时 → 复用 width chunk 实现；按用户使用固定宽度布局，测试 chunk/fullbatch 与用户映射一致，不将 ragged candidates 误拼为不同用户。
- 日志指标重新按分数排序、trace 仍标为 content 会误报 → 返回实际最终 score，严格协议区分并验证真实输出；保留独立原始输出复算门禁。

## Migration Plan

实现相关 hook、评分复用、v4 配置参数和 trace validator；完成初始化/default 回归、概率恒等式、label independence、同候选配对、严格契约及旧 scoring 回归。主代理补充 Hydra 配置与单进程命令、验证 OpenSpec、同步实际源码并执行一次 evaluation。content 默认提供回退。

## Open Questions

新 v4 混合终排能否优于同候选 content 尚未验证。evaluation 晋级不等于 Testing 达到相对 LIGER dense 10% 的全线程目标。
