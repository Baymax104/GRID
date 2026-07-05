## Context

当前官方数据加载配置仍暴露 `assign_all_files_per_worker`，并在量化训练实验中显式开启该策略。该能力允许每个 dataloader worker 读取完整文件集合，再依赖 worker 内部切分与 shuffle 近似实现训练期采样覆盖。但基于仓库现有 Amazon beauty / toys / sports 数据分片情况，这一策略不再是主线实验的必要前提，反而增加了 datamodule、dataset、配置与注释层面的理解成本。

用户已经明确希望先收缩配置面：删除该配置项及相关逻辑，使后续新增实验时不必再判断该开关是否需要存在。

## Goals / Non-Goals

**Goals:**
- 删除 `assign_all_files_per_worker` 的官方配置语义。
- 删除 datamodule / dataset / worker 文件分配中围绕该开关的专门逻辑。
- 更新当前量化训练 experiment，移除该配置项。
- 保持标准文件分摊语义下的训练/推理主链路可继续装配。

**Non-Goals:**
- 不重新设计新的 worker 采样策略替代该开关。
- 不调整数据文件切分格式或现有 TFRecord 内容。
- 不对 `assign_files_to_workers()` 的常规文件平衡策略做额外优化。

## Decisions

### 1. 彻底删除 `assign_all_files_per_worker` 配置能力
- 决策：从 dataloader config 类型、experiment YAML 与主链路实现中移除该字段。
- 原因：对于当前官方数据集与实验，该字段不是必要能力，继续保留只会制造额外认知成本。
- 备选方案：保留字段但默认 false。未采用，因为用户目标是减少配置面而不是仅调整默认值。

### 2. 统一回到 worker 间标准文件分摊
- 决策：`assign_files_to_workers()` 与 dataset worker 逻辑仅保留按 worker 切分文件的常规路径，不再支持“全部 worker 拿全部文件”。
- 原因：这让文件分配语义更直接，也避免训练与非训练阶段在同一开关上承载特殊含义。
- 备选方案：仅删除实验配置，底层逻辑保留。未采用，因为会留下不可见但仍存在的历史分支。

### 3. 删除 item datamodule 中的专属约束判断
- 决策：随着该配置项被删除，`ItemDataModule` 中围绕非训练阶段禁止该策略的校验逻辑一并删除。
- 原因：约束只服务于已删除能力，继续保留会形成死代码。

## Risks / Trade-offs

- [未来若回到“文件数远小于 worker 数”的极端训练场景，可能怀念该策略] → 接受这是官方能力收缩；如未来确有需要，再基于真实场景重新设计。
- [文档/注释可能残留历史表述] → 实现后做全文搜索，清理核心代码、配置与关键说明文档。
- [删除后若量化训练吞吐或效果变化，定位需要额外观察] → 本轮先以结构收缩为目标，后续若有性能问题再基于实验数据复核。

## Migration Plan

1. 从 dataloader config 类型中删除 `assign_all_files_per_worker` 字段。
2. 删除 datamodule、dataset 和 worker 文件分配函数中对该字段的处理分支。
3. 从量化训练 experiment 配置中移除该字段。
4. 做全文搜索与最小静态检查，确认官方实验不再声明或依赖该配置能力。

## Open Questions

- 当前无阻塞性开放问题。
