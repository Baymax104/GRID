## Context

Tail-SID diagnosis 通过统一 `src.main` 入口和 Lightning `Trainer.test` 运行。当前 `DiagnosisDataModule.setup_stage` 将原始 `semantic_id_path` 与可选 `embedding_path` 直接传给 `DiagnosisDataset`，Dataset 在首个 `__getitem__` 中调用 `load_model_output`。本地路径可以直接加载，但短 W&B URI 缺少显式 identity；同时默认 `field_name=model_output_path` 会把 Semantic ID 或 embedding 错误识别为 `recommendation_output`。

项目已经规定短 W&B URI 的 identity 只能来自实验配置显式传入的 `user/project`，不得从环境变量、活动 run 或 W&B 账户默认值推断。Artifact role 应由字段名推断，底层解析与下载必须继续归 `src/data/components/artifacts.py` 和 `src/utils/wandb.py`。

Lightning 2.6.5 在 strategy environment 建立后先调用 `datamodule.setup`，再调用 callback `setup`。现有 lineage callback 会在 callback setup 阶段读取 resolved-reference registry；若继续把解析延迟到首个 Dataset batch，diagnosis 输入会错过该记录时机。

## Goals / Non-Goals

**Goals:**

- 让 diagnosis 的短 Semantic ID 与 embedding W&B URI 使用实验 `${user}/${project}` 成功解析。
- 使用 `semantic_id_path` 与 `embedding_path` 字段语义推断正确 Artifact role。
- 在分布式 strategy environment 内完成 W&B Artifact 下载协调。
- 在 callback setup 前注册 resolved references，使现有 lineage callback 能记录 upstream Artifact。
- 保持本地路径、完整跨项目 URI、可选 embedding 和 keyed bundle 内容协议兼容。
- 保持 `DiagnosisDataset` 的路径输入及 keyed bundle 解析职责不变。

**Non-Goals:**

- 不修改 W&B URI grammar、Artifact 选择规则、下载 helper 或 logger 生命周期。
- 不恢复环境变量、活动 run 或 W&B 默认账户的 identity fallback。
- 不修改 launcher，也不让 launcher识别领域路径字段。
- 不修改 `merged_predictions_tensor.pt` 的 `{"keys", "predictions"}` 协议。
- 不把 diagnosis 输入改为预加载的 `ModelOutput`，也不引入新的 loader/config 数据类。
- 不修改 diagnosis 指标、训练频率分组或报告语义。

## Decisions

### 1. 在 diagnosis test dataloader config 中显式映射 W&B identity

`configs/data/tail_sid_diagnosis.yaml` 将在 `test_dataloader` 下声明 `wandb_entity: ${user}` 与 `wandb_project: ${project}`。这与量化和 TIGER Artifact loader 的参数命名保持一致，同时保留实验入口作为 identity 的单一来源。

未采用新的顶层 `entity`、`wandb` namespace 或环境变量 fallback，因为当前显式 identity 设计已经确定 `user/project` 是官方实验的配置契约。

### 2. 在 `DiagnosisDataModule.setup_stage` 中预解析引用到本地路径

DataModule 在构造 Dataset 前分别调用统一 `resolve_reference`：

- Semantic ID 使用 `field_name="semantic_id_path"`；
- 非空 embedding 使用 `field_name="embedding_path"`；
- 两者都传入 config 中的 `wandb_entity/wandb_project`；
- embedding 为 `None` 时不调用 resolver。

解析后的本地路径继续传给 `DiagnosisDataset`。本地输入由 resolver 原样返回，不触发 W&B API。

选择 DataModule setup 而不是 Dataset 首个 batch，是因为此时 Lightning strategy environment 已建立，分布式下载 helper 可以执行 rank-zero 下载与 barrier；同时 callback setup 尚未执行，resolved-reference registry 对 lineage callback 可见。

### 3. 只预解析路径，不在 DataModule 中预加载 bundle

DataModule 不直接调用 `load_model_output` 生成 `ModelOutput`，而只解析 URI。Dataset 继续通过 `load_model_output(local_path)` 完成 bundle 读取、校验、SID views 构造和 embedding key lookup。

这一选择避免改变 `DiagnosisDataset` 构造契约和现有本地测试，也避免同时支持 path/bundle 两套输入。Artifact role 与 lineage 已在前置解析阶段确定，Dataset 对本地路径的默认 `model_output_path` 不再触发 W&B role 推断。

未采用 Hydra eager bundle instantiation，因为 datamodule 在 launcher 中先于 Trainer 实例化，届时分布式 environment 尚未建立，可能造成每个 rank 独立下载，并且会扩大配置与 Dataset 契约变更。

### 4. 复用现有 resolver、registry 和 lineage callback

DataModule 只调用 `src.data.components.artifacts.resolve_reference`，不得实现 URI parsing、W&B API 查询、Artifact 选择或下载。现有 resolver 继续负责 registry 写入；现有 `WandbArtifactLineageCallback` 继续负责 `use_artifact`，无需新增 diagnosis 专用 callback 或 hook。

### 5. 测试同时锁定 identity、role、时序和兼容性

Hydra compose 测试验证 diagnosis test dataloader 获得实验 identity。DataModule 单元测试 mock resolver，验证两个字段名、identity、可选 embedding 行为以及解析后路径转发。组合测试验证 DataModule setup 后 callback setup 可以读取 registry。现有本地 bundle、shape、key lookup 和指标测试继续保留。

## Risks / Trade-offs

- [Risk] DataModule 开始承担输入引用解析职责。→ Mitigation：只调用 data 层统一 resolver，不实现 W&B 逻辑；Dataset 仍承担 bundle 内容读取与领域转换。
- [Risk] setup 阶段解析失败会使任务早于首个 batch 终止。→ Mitigation：这是预期的 fail-fast 行为，能在指标计算前暴露 identity、role、Artifact 缺失或歧义错误。
- [Risk] 未来若 callback setup 顺序变化，lineage 时序假设可能失效。→ Mitigation：增加 DataModule setup 与 callback setup 的聚焦组合测试，并依赖当前锁定的 Lightning 2.6.5。
- [Risk] 多个 Artifact 使用同一 identity，但可能来自不同项目。→ Mitigation：完整 `wandb://<entity>/<project>/<run-id>` URI 继续覆盖 experiment defaults。
- [Risk] 新 change 与尚未归档的 `require-explicit-wandb-entity` 修改相邻能力。→ Mitigation：delta spec 使用显式 identity 的最终契约，并在归档时按依赖语义先合并 identity change、再合并本 change。

## Migration Plan

1. 先完成并归档 `require-explicit-wandb-entity`，使 living specs 包含显式 identity 契约。
2. 更新 diagnosis data config、DataModule 与聚焦测试。
3. 运行 Hydra compose、聚焦 pytest、Ruff 和本 change 的严格 OpenSpec 校验。
4. 使用短 Semantic ID/embedding URI 做显式 dry-run 或最小 diagnosis 验证。
5. 若需回滚，只需恢复 diagnosis config 与 DataModule 装配；本地数据、bundle 文件和 W&B Artifact 均无需迁移。

## Open Questions

无。修复范围、identity 来源、role 映射、解析时机和兼容策略均已确定。
