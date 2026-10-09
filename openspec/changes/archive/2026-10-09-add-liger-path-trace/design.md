## Context

CoPMRec 使用 Hugging Face generate 和 ProbabilityMixtureProcessor；candidate trace 是严格的 v1 schema。路径信息必须来自解码时实际保留的前缀，不能从最终序列反推中途存活。

## Goals / Non-Goals

支持同 checkpoint legal/max/mass 的目标路径存活、首次丢失层及条件概率比较。不改变搜索，不新增模型参数，不自动启动实验，不复原旧产物不存在的路径。

## Decisions

- 包装现有 processor 并原样返回其输出；下一步的输入就是上一层实际保留的 beam，最后一层取 generate 返回的完整序列。记录全部 beam 前缀（剩余深度填 -1）。不更换 beam 实现，不依赖 HF 私有 beam scorer。
- 仅当目标父前缀仍在当前 frontier 时记录目标下一分支的生成、内容和混合 log probability、合法子分支数量及父节点内竞争排名（严格大于的数量加一，非跨 beam 全局排名）。父前缀已丢失时使用 NaN 与显式可达掩码；不做 teacher-forcing 反事实补算。
- 使用独立 liger_paths_v1 payload 和 AuxiliaryTensorWriter，candidate v1 保持不变。字段包括 beam_prefixes、target_prefix_survived、target_parent_present、first_failure_depth 和分支统计。独立 validator 检查形状、前缀祖先、存活单调性及缺失语义。
- 开关默认关闭，只允许 probability_mixture、hybrid 与 candidate_trace；新回调组合保留 prediction、candidate 和 lineage writer，新增本地引用路径 writer。

## Risks / Trade-offs

- 观察增加 CPU 复制及条件概率计算 → 仅显式开启；不用于效率比较。
- 不同 Transformers 的 frontier 行布局可能变化 → 使用真实微型模型，三种控制及多 batch/beam 验证输出不变与最终生成命中一致。
- 旧 mass 没有路径 → 三臂命令均启用新 trace；mass 路径采集需用户手动启动并与原 mass 预测逐用户核对。
