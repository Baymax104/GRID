## Context

`KMeansLayer` 目前通过 `init_buffer` 收集样本，在 buffer 满后由 rank 0 计算 `init_centroids`，再返回初始化 loss，让 manual optimization 在当前 step 将 `centroids` 更新到初始化目标；下一次 `train_step()` 才通过 `is_initial_step` 将 layer 标记为 initialized。这个设计保持了旧 DDP 初始化路径，但状态模型复杂：维护者需要同时理解 `is_initial_step`、`is_initialized`、`init_centroids` 与 `init_buffer` 的跨 step 关系。

用户明确倾向于简单状态，并选择“rank 0 初始化 + broadcast”的方案。因此本设计将初始化改为直接写入 centroids，并通过 distributed broadcast 保证多卡一致性。

## Goals / Non-Goals

**Goals:**

- 将 RKMeans `KMeansLayer` 初始化状态简化为单一 initialized 布尔语义。
- 删除 `is_initial_step` 过渡状态与 `init_centroids` 跨 step 暂存。
- `init_buffer` 满后直接用 K-Means++ 结果初始化 `centroids`。
- DDP 场景下 rank 0 负责初始化，所有 rank 通过 broadcast 获得相同 centroids。
- 保持 `ResidualKMeans` 的 layer-wise schedule、checkpoint initialized 状态、Hydra 配置接口和输出协议不变。

**Non-Goals:**

- 不改变 RKMeans 的逐层训练 schedule。
- 不改变 RQVAE/RVQ 初始化逻辑。
- 不引入跨模型共享初始化抽象。
- 不更改训练脚本或 experiment 配置字段。
- 不恢复旧 checkpoint 兼容性。

## Decisions

### Decision 1: 用一个 initialized 状态替代两个初始化布尔状态

`KMeansLayer` SHALL 删除 `is_initial_step`。外部只通过 `layer.is_initialized` 或等价 property 判断该层是否完成初始化。

Rationale:

- 当前两个 bool 实际编码一个三态流程，理解成本高。
- 直接初始化 centroids 后，不再需要“初始化 loss 已产生但 optimizer step 尚未完成”的中间态。
- `ResidualKMeans` 的调度语义天然只需要知道当前 layer 是否可进入正常训练。

### Decision 2: 初始化完成时直接写入 centroid 参数

当 `init_buffer` 达到 `init_buffer_size` 时，`KMeansLayer` SHALL 计算 initial centroids，并在 `torch.no_grad()` 下执行 `self.centroids.copy_(initial_centroids)`。

Rationale:

- 初始化是参数赋值，不需要通过 gradient loss 间接表达。
- 去掉初始化 loss 后，`train_step()` 的返回语义更简单：未初始化且 buffer 未满时返回 zero loss；初始化完成后返回基于新 centroids 的 ids/embeddings 与零或正常更新 loss。

### Decision 3: DDP 下 rank 0 初始化并 broadcast centroids

`KMeansLayer` SHALL 提供 quantization-local helper，检测 `torch.distributed.is_available()` 与 `torch.distributed.is_initialized()`。若处于 distributed 环境：

- rank 0 使用本地 `init_buffer` 计算 K-Means++ centroids；
- 非 rank 0 使用同 shape zero tensor 作为接收缓冲；
- 所有 rank 对该 tensor 调用 `torch.distributed.broadcast(..., src=0)`；
- 所有 rank 将 broadcast 后的结果 copy 到 `self.centroids`。

非 distributed 环境直接在当前进程计算并 copy。

Rationale:

- 相比每 rank 各自初始化，broadcast 保证所有 rank 初始参数一致。
- 相比继续使用初始化 loss/manual optimizer step，broadcast 更直接，状态更少。

### Decision 4: `init_buffer` 仍为非持久运行期状态

`init_buffer` 仍用于收集初始化样本，不进入 checkpoint；初始化完成后清空。`cluster_counts` 仍为 `persistent=False` buffer，并在训练开始重置。

Rationale:

- 推理只需要 centroids，不需要初始化样本或 online update counts。
- 运行期状态不应扩大 checkpoint contract。

## Risks / Trade-offs

- [DDP broadcast 调用错误] 未初始化 distributed 环境时调用 broadcast 会失败 → 使用 helper 先检查 distributed availability/initialization。
- [非 rank 0 无 init data] 非 rank 0 不参与 K-Means++ 初始化，初始化质量只取决于 rank 0 buffer → 这是方案 2 的明确取舍，换取确定的一致 centroid；后续可用 all-gather 扩展，但本次不做。
- [初始化 step loss 语义变化] 初始化完成的 step 不再产生初始化 loss → 单元测试覆盖 buffer 未满、初始化完成、初始化后更新；训练 loop 的 `is_initialized` 参数应在初始化完成后立即为 true。
- [checkpoint runtime state 简化] 旧运行中 checkpoint 若依赖 `is_initial_step` 无法恢复该过渡状态 → 用户已接受结构性简化；新的 checkpoint 只保存 initialized 状态。
