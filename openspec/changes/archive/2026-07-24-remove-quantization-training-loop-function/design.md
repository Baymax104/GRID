## Context

当前 quantization 训练配置通过 `training_loop_function` 注入 `scale_loss_by_world_size_for_initialization_training_loop`。该函数把模型切到 Lightning manual optimization，用于在 DDP 初始化阶段将 rank 0 的 init loss 乘以 `world_size`，再用临时 `SGD(lr=0.5)` 将 centroids 一步更新到目标初始化值。

这条路径的复杂度来自旧初始化策略：初始化结果不是直接写入 centroids，而是通过 loss 和 optimizer 间接搬运。RKMeans 已经改为 rank 0 计算 centroids、broadcast、`copy_` 到参数；RVQ/RQVAE 仍保留旧 init loss 路径。因此要完全删除 custom loop，需要先把 RVQ/RQVAE 也改为 direct centroid assignment。

## Goals / Non-Goals

**Goals:**

- 删除 quantization model config 中的 `training_loop_function` 字段。
- 删除三个量化模型 constructor 中的 `training_loop_function` 参数和 manual optimization 分支。
- 删除 `src/quantization/training_loop_functions.py`，以及 `scale_loss_by_world_size_for_initialization_training_loop` 的所有 active 引用。
- 将 RVQ/RQVAE 初始化改为 rank-zero compute + broadcast + direct `copy_`。
- 让 RKMeans/RVQ/RQVAE 使用 Lightning 默认 automatic optimization。

**Non-Goals:**

- 不重写量化算法本身、residual traversal、centroid 更新 loss、step budget 或指标。
- 不改变 checkpoint centroid tensor 布局、推理输出 bundle、训练脚本命令形态。
- 不引入新的 Trainer/Callback 抽象。
- 不把 RKMeans/RVQ/RQVAE 合并到共享基类。

## Decisions

### Decision 1: 删除 custom loop，而不是迁移到 Callback 或 Trainer

`training_loop_function` 只服务于旧的初始化参数搬运方式。direct centroid assignment 后，不再需要 manual backward、临时 optimizer、scheduler 手动 step。保留 Callback/Trainer 版本会制造新的外部控制面。

Alternative considered: 将函数改成 Callback。该方案仍把量化初始化细节暴露给训练编排层，和模型自包含目标冲突。

### Decision 2: 初始化统一为 direct assignment

RVQ/RQVAE 在 buffer 满后直接计算初始化 centroids，将结果 broadcast 到所有 rank，并在 `torch.no_grad()` 下 `copy_` 到对应 layer centroids。初始化 step 返回零 loss 或不产生可训练初始化 loss，下一次正常训练 step 由 Lightning 默认 optimizer 处理。

Alternative considered: 仅删除配置字段，保留 Python manual optimization 可选路径。该方案会留下死代码，并且外部调用仍可触发旧路径。

### Decision 3: 保留 rank-zero 作为初始化来源

DDP 下不同 rank 的 buffer 不完全相同。为了避免每张卡得到不同 centroids，继续由 rank 0 计算初始化目标，再通过 `torch.distributed.broadcast` 同步。

Alternative considered: 每个 rank 各自初始化。该方案更简单，但会破坏 DDP 参数一致性。

### Decision 4: RVQ/RQVAE 清理初始化过渡状态

direct assignment 后，`is_initial_step_list`、`init_centroids_list`、`init_loss_function` 这类跨 step 的初始化搬运状态不再必要，应删除或收敛到单一 `is_initialized_list` 状态。

## Risks / Trade-offs

- **Risk: RVQ/RQVAE 初始化时机变化** → Mitigation: 初始化仍在 buffer 满的同一个训练 step 发生，但从“下一步标记 initialized”改为“同一步直接 initialized”；通过 unit tests 覆盖状态变化。
- **Risk: DDP broadcast helper 处理不一致** → Mitigation: 复用 RKMeans 中已验证的 rank/initialized/broadcast 模式，测试 rank 0 与非 0 行为。
- **Risk: scheduler 行为变化** → Mitigation: 删除 manual optimization 后 Lightning 自动按 `configure_optimizers()` 返回的 step scheduler 行为执行，不再需要 custom loop 中手动 `scheduler.step()`。
- **Risk: 外部 override 仍传 `training_loop_function`** → Mitigation: 这是 breaking config cleanup；实现后仓库内 active config/code 无残留，外部 override 需删除该字段。

## Migration Plan

1. 从 RKMeans/RVQ/RQVAE 训练 YAML 删除 `training_loop_function` 字段和 `_target_` block。
2. 从三种模型 constructor 删除 `training_loop_function`，删除 `automatic_optimization=False` 和 `training_step()` 中 custom loop 调用。
3. RVQ：实现 rank-zero direct K-Means++ initialization + broadcast + `copy_`，删除 init loss 搬运状态。
4. RQVAE：保留 K-Means++ + convergence 算法，但将最终 centroids broadcast 后直接 `copy_`，删除 init loss 搬运状态。
5. 删除 `src/quantization/training_loop_functions.py`。
6. 更新/新增 tests，运行 targeted pytest 和 Hydra target smoke check。

Rollback: 若需要恢复特殊初始化 optimizer，应通过新 change 明确重新引入，并补充 DDP 行为测试。

## Open Questions

无。
