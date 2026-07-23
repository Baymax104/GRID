## Context

RKMeans、RVQ、RQVAE 都仍暴露 `initialize_on_cpu: bool = False`，并在 K-Means++ 初始化路径中把候选样本临时移到 CPU 后再移回原 device。这个分支的历史目的主要是降低初始化阶段 GPU 显存压力。

当前可执行配置中该字段全部为 `false`，实际训练/推理 pipeline 一直走当前 device 初始化路径。继续保留该参数会把一个未使用的实现细节暴露给 Hydra model config，同时增加三个量化模型初始化逻辑的分支数量。

## Goals / Non-Goals

**Goals:**

- 从 RKMeans / RVQ / RQVAE 的 public constructor 和 Hydra model config 中删除 `initialize_on_cpu`。
- 删除 K-Means++ 初始化中的 CPU 分支，使初始化始终使用当前 tensor device。
- 保持现有默认运行行为不变，因为所有现行配置本来就是 `initialize_on_cpu: false`。
- 更新测试与最小 smoke check，确认相关 model target 仍可解析。

**Non-Goals:**

- 不重新设计 K-Means++ 算法或 centroid 更新规则。
- 不改变量化模型 checkpoint 中 centroids 的保存结构。
- 不引入新的显存优化策略或 fallback 机制。
- 不修改训练脚本、pipeline 顺序、输出 bundle 格式。

## Decisions

### Decision 1: 删除参数，而不是保留 deprecated no-op

`initialize_on_cpu` 是构造参数和配置字段；若保留 no-op，会继续让用户误以为存在可用的 CPU 初始化策略。因此本变更直接删除参数和配置字段。

Alternative considered: 保留参数但忽略它。该方案兼容旧 override，但无法达到简化 public config contract 的目标。

### Decision 2: K-Means++ 初始化始终在当前 device 上执行

RKMeans `KMeansLayer`、RVQ、RQVAE 的 `_kmeans_plus_plus_init(...)` 或等价调用不再接收 `initialize_on_cpu`。初始化输入 buffer 保持在当前训练 device 上，返回 centroids 也保持同一 device。

Alternative considered: 仅删除 YAML 字段、保留 Python 参数。该方案会留下未使用分支，后续维护者仍需理解 CPU 路径。

### Decision 3: 同步清理三种量化模型

虽然用户问题从 RKMeans 触发，但同名参数同时存在于 RVQ 和 RQVAE，且所有配置均为 false。为了避免同一概念在不同量化模型中不一致，本变更一次性删除三处。

Alternative considered: 只清理 RKMeans。该方案范围更小，但会让 RVQ/RQVAE 继续暴露同样的无效开关。

### Decision 4: 将旧 override 视为 breaking config change

删除 constructor 参数后，外部 Hydra override 若仍传入 `model.initialize_on_cpu=...` 或 YAML 字段残留，会触发实例化错误。这是有意的 contract 收紧。

## Risks / Trade-offs

- **Risk: 某些本地未提交配置仍传入 `initialize_on_cpu`** → Mitigation: 在任务中全局搜索并删除仓库内残留；最终说明这是 breaking config cleanup，本地 override 需同步移除。
- **Risk: 大规模初始化曾依赖 CPU fallback 避免 GPU OOM** → Mitigation: 当前仓库没有启用该路径的可执行配置；若未来需要显存优化，应以明确的新策略重新设计，而不是保留未验证开关。
- **Risk: 三个模型的 helper 签名不一致导致遗漏** → Mitigation: 对 `initialize_on_cpu` 做全局搜索，并运行针对量化 model target 的导入/解析 smoke check。

## Migration Plan

1. 删除四个 model YAML 中的 `initialize_on_cpu: false`。
2. 删除 RKMeans / RVQ / RQVAE constructor 参数与属性。
3. 删除 K-Means++ helper 的 `initialize_on_cpu` 参数和 CPU 分支。
4. 更新调用点与测试。
5. 运行 `uv run pytest tests/quantization/rkmeans/test_kmeans_layer.py` 以及 Hydra target/import smoke check。

Rollback: 如确实需要恢复 CPU 初始化路径，可通过新 OpenSpec change 重新引入带明确使用场景和测试的显存优化策略。

## Open Questions

无。
