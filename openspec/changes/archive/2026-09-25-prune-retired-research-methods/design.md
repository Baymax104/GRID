## Context

当前仓库同时包含现行 CoPMRec/LIGER 路线、TIGER 基线、上游 embedding/quantization 流水线，以及 BRIR、MIR、CGBS 和多轮已结题 LIGER 变体。大量历史模块仍通过 Hydra `_target_`、根目录脚本、writer 和测试形成可执行表面；其中部分代码尚未提交，因此不能用 Git 跟踪状态判断是否活跃。研究状态已冻结主方法为 mass 前缀聚合，max 与 legal generation 仅保留为机制控制。

## Goals / Non-Goals

**Goals:**

- 只保留 CoPMRec、LIGER、TIGER、embedding、quantization、candidate trace 与共享运行基础设施。
- 删除已结题方法的完整垂直切片，包括实现、配置、脚本、writer 和专用测试。
- 将固定概率混合放回 LIGER 核心，保持现有 checkpoint 参数名和主实验命令兼容。
- 通过静态引用检查、Hydra compose、聚焦单测和全量测试确认没有悬空入口。

**Non-Goals:**

- 不删除历史论文记录、OpenSpec 归档、W&B runs 或 Artifact。
- 不改变当前训练预算、模型超参数、数据协议或指标定义。
- 不启动训练、推理或完整实验。

## Decisions

### 1. 按垂直切片删除，而不是保留兼容壳

BRIR、MIR/item-resolution、CGBS/catalog-grounded 和 training probe 的实现、配置、脚本与测试一起删除。兼容壳会继续暴露失效入口并扩大维护表面，违反清理目标。

### 2. 保留三层活跃方法结构

`src/recommendation/tiger/` 保留基础生成推荐，`src/recommendation/liger/` 保留 LIGER 与 CoPMRec，embedding/quantization 保持独立领域目录。方法专用数据预处理仍位于 `src/data/components/liger.py`，跨实验 Artifact 解析继续位于 `src/data/components/artifacts.py`。

### 3. 固定混合不依赖动态门控模块

把数值稳定的两路 log-probability 混合函数放入 `candidate_guidance.py`。`JointMixtureLiger` 保留 `dynamic_gate.bias` 参数名以加载既有 checkpoint，但移除动态特征、外部 gate checkpoint 和训练后 gate adapter。

### 4. 当前机制控制限定为 legal、mass、max

删除 `max_root_mass_deep`、preference dispersion、candidate union 和 source protection。保留 `legal_generation` 与 `max_mixture`，因为论文实验矩阵仍明确使用它们。

### 5. 历史规范保留，活跃 change 只记录当前清理

旧 OpenSpec change 和研究文档作为决策历史保留，不再对应当前可执行代码。新结构测试扫描现行 `src/`、`configs/` 和启动脚本，防止已淘汰入口回流。

## Risks / Trade-offs

- [既有 checkpoint 不能加载] → 保留 CoPMRec 的 `dynamic_gate.bias` state-dict 路径，并用合成 checkpoint 回归测试验证。
- [误删共享 helper] → 删除前以 import/Hydra `_target_` 反向搜索，删除后执行静态引用扫描和全量测试。
- [历史命令失效] → 历史文档继续保留结果和 Git/W&B 身份，但明确当前仓库只复现活跃论文矩阵。
- [工作区已有未提交修改] → 只修改或删除已被研究状态判定为淘汰的垂直切片；其余修改保持原样。

## Migration Plan

1. 先收缩 LIGER 核心并通过核心测试。
2. 删除 LIGER 已结题变体的垂直切片。
3. 删除 BRIR、MIR、CGBS 和 training probe 垂直切片。
4. 清理 Artifact loader 与残留引用，增加结构回归测试。
5. 执行 Hydra compose、脚本语法、聚焦测试、全量测试和 OpenSpec strict 验证。

## Open Questions

无。当前保留/删除边界由 `research-state.yaml` 与 `current-plan.md` 的冻结路线决定。
