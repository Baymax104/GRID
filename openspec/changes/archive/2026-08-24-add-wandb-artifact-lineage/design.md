## Context

当前 GRID 阶段链路依赖本地文件路径传递产物：训练阶段 checkpoint 写入 `${paths.output_dir}/checkpoints`，推理阶段 `LocalPickleWriter` 将 keyed prediction bundle 合并为 `${paths.output_dir}/pickle/merged_predictions_tensor.pt`，下游通过 `ckpt_path`、`embedding_path`、`semantic_id_path` 等字段读取这些路径。训练实验已经使用 W&B logger 记录 metrics、config 和 notes，但训练 checkpoint 与推理产物没有标准化为 W&B Artifact，W&B 中也无法自动展示 `train -> inference -> train -> inference` 的产物 lineage。

用户目标是保持现有参数名与本地路径兼容，同时允许短引用，例如 `semantic_id_path=wandb://<run-id>` 和 `ckpt_path=wandb://<run-id>`。实现必须遵守模块边界：W&B URI 解析和 Artifact 下载属于 data 读取职责，Artifact 发布属于 writer 职责，lineage 记录属于 callback 职责；未使用 W&B 引用或发布配置时不触发 W&B；logger、writer、callback 不应形成配套绑定关系。

## Goals / Non-Goals

**Goals:**
- 让 `ckpt_path`、`embedding_path`、`semantic_id_path` 等现有字段支持 `wandb://<run-id>`，并解析到具体 Artifact 文件。
- 让训练和推理阶段可选发布 output Artifact，并让下游阶段通过 `use_artifact` 记录 W&B lineage。
- 保持普通本地路径、已有 `LocalPickleWriter`、现有 W&B logger 配置的行为兼容。
- 将 W&B 引用解析和 Artifact 下载放在 data 读取职责内，将 Artifact 发布放在 writer 职责内，并由配置显式启用。

**Non-Goals:**
- 不改变 keyed prediction bundle 文件内容协议。
- 不要求所有实验必须启用 W&B。
- 不把 `LocalPickleWriter` 改造成 W&B writer。
- 不用 `latest` 作为默认复现实验引用策略。
- 不引入新的训练、推理或分析主入口。

## Decisions

### Decision 1: 扩展现有 path 字段的 value protocol

`ckpt_path`、`embedding_path`、`semantic_id_path` 继续表示“当前阶段要读取的文件引用”，但其值可以是本地路径、fsspec 支持的远端路径，或 `wandb://` URI。字段名用于推断默认 Artifact role：

| 字段 | 默认 role | 默认文件 |
|---|---|---|
| `ckpt_path` | `checkpoint` | best checkpoint 文件 |
| `embedding_path` | `semantic_embedding` | `merged_predictions_tensor.pt` |
| `semantic_id_path` | `semantic_id` | `merged_predictions_tensor.pt` |

默认短格式为 `wandb://<run-id>`，跨项目格式为 `wandb://<entity>/<project>/<run-id>`。URI 查询参数可覆盖 role、alias、file，例如 `wandb://abc123?role=semantic_id&alias=v3&file=merged_predictions_tensor.pt`。

备选方案是新增 `semantic_id_run_id`、`ckpt_run_id` 等字段。该方案更显式，但会扩大脚本和配置面，且用户仍需记住字段组合。扩展现有 path 字段更符合兼容目标。

### Decision 2: Artifact lineage 基于 W&B Artifact，而不是 run 文件列表

`wandb://<run-id>` MUST 解析 producer run 的 output Artifact，而不是扫描 run files。解析成功后，当前 W&B run 若存在，则调用 `use_artifact` 记录输入 lineage。这样 W&B 能展示 producer run、Artifact、consumer run 的关系。

备选方案是从 run 文件列表下载 `merged_predictions_tensor.pt` 或 checkpoint。该方案短期简单，但没有 Artifact 版本、alias、metadata 和 lineage，不满足实验链路可追踪目标。

### Decision 3: data/components/artifacts.py 承载 artifact 读取协议

新增 `src/data/components/artifacts.py` 负责：
- 承载或迁移 `load_model_output`、`load_semantic_id_tensor` 等 artifact/bundle 加载函数。
- 基于字段名推断默认 role 和默认文件。
- 调用 `src/utils/wandb.py` 中的 W&B URI parser、Artifact selector/downloader，把 `wandb://` 引用转换为具体本地文件路径。
- 记录已解析输入引用，供显式 lineage callback 使用。

`load_model_output`、`load_semantic_id_tensor` 这类函数天然负责“把输入引用变成可读取数据”，因此字段级 resolver 位于 data component；纯 W&B API helper 放在 `src/utils/wandb.py`，让 `artifacts.py` 保持清晰。resolver 对非 `wandb://` 值直接透传；`wandb://` 值才延迟导入 W&B backend。launcher 不扫描配置、不判断是否存在 W&B 引用、不为了 Artifact lineage 提前实例化 logger。

### Decision 4: writer 能力收敛到 common/writers

删除旧 `src/common/inference/`，将 writer 能力整理到 `src/common/writers/`。`LocalPickleWriter`、`WandbArtifactWriter` 与 checkpoint writer 在该目录下平级，并分别放在 `local_pickle_writer.py`、`wandb_artifact_writer.py`、`wandb_checkpoint_writer.py`。`LocalPickleWriter` 负责本地 bundle 写入，`WandbArtifactWriter` 负责把已经存在的最终产物发布到 W&B Artifact。`WandbArtifactWriter` 不参与 batch flush、rank-local shard 合并或 bundle 内容生成。

训练 checkpoint 不是 prediction bundle，因此使用 `src/common/writers/wandb_checkpoint_writer.py`，而不是塞入 `src/utils` 或与 prediction bundle writer 混放。

checkpoint writer 不独立保存 checkpoint，也不复制 `ModelCheckpoint` 的 monitor、mode、save_top_k、filename 等决策。它只在 Lightning `ModelCheckpoint` 完成写入后读取其产物并发布为 W&B Artifact。也就是说，W&B checkpoint Artifact 的内容会跟随当前 `ModelCheckpoint` 配置变化：例如 TIGER 当前发布 `val/recall@10` 下的 best checkpoint，而 RQ-VAE/RVQ/RKMeans 当前发布 `train/loss` 下的 best checkpoint。该影响是预期行为，因为本地 checkpoint 仍是训练阶段的单一事实来源。

如果一个实验没有启用 `ModelCheckpoint`，或者启用了多个 checkpoint callback，checkpoint writer 必须显式处理：无 checkpoint 时按配置选择 skip 或 fail；多个 checkpoint callback 时必须通过配置指定 selection policy，不能静默猜测。

### Decision 5: logger、writer、lineage callback 低耦合

W&B logger 只负责 metrics/config/notes logging；本地 writer 只负责本地落盘；W&B writer 只负责 Artifact 发布；`WandbArtifactLineageCallback` 只负责把 resolver 记录的上游 Artifact 标记为当前 run 使用。四者之间不建立强制绑定：
- 使用 W&B logger 不自动要求启用 W&B writer。
- 使用 `LocalPickleWriter` 不自动要求启用 W&B。
- 使用 `WandbArtifactWriter` 不要求替换 `LocalPickleWriter`；如果它要发布本地 writer 的输出，只通过配置读取最终文件路径。
- 使用 `wandb://` 读取上游 Artifact 不要求启用 W&B output writer。

### Decision 6: lineage 记录由显式 callback 完成

`src/data/components/artifacts.py` 中的 resolver 成功解析 `wandb://` 后，将原始引用、producer run、Artifact 标识和本地缓存路径写入进程内 resolved-reference registry。显式配置的 `WandbArtifactLineageCallback` 位于 `src/common/callbacks/`，在 W&B run 已经存在时读取 registry 并调用 `use_artifact`。如果未配置该 callback，`wandb://` 仍可下载并供本地流程使用，但不会隐式创建 run 或改变 logger 初始化顺序。

### Decision 7: 解析结果写回可审计 metadata

解析 `wandb://` 后，resolved metadata 应保留原始引用、解析出的 Artifact name/version、producer run id、local cache path。该 metadata 由 `src/data/components/artifacts.py` 提供给 `WandbArtifactLineageCallback` 或可选 config logger，避免在 launcher 中直接改写 Hydra 主配置结构。

## Risks / Trade-offs

- **Risk:** 一个 producer run 产出多个同 role Artifact，短 URI 可能歧义。→ **Mitigation:** resolver 必须要求唯一匹配；不唯一时报错并提示补充 `alias`、`role` 或 `file`。
- **Risk:** 使用 `latest` 会让复现实验漂移。→ **Mitigation:** 默认优先解析固定版本或 run output 中唯一 Artifact；脚本示例推荐 run id + artifact version，而非 `latest`。
- **Risk:** W&B 网络或凭据不可用会阻塞使用 `wandb://` 的实验。→ **Mitigation:** 仅 `wandb://` 或 Artifact 发布配置触发 W&B；错误信息明确指出认证、项目、run 或 Artifact 问题。本地路径流程不受影响。
- **Risk:** 上传大型 `.pt` 文件增加推理结束耗时。→ **Mitigation:** `WandbArtifactWriter` 是可选 writer；本地 writer 先完成本地产物，上传失败不破坏已经落盘的本地文件，失败策略由 writer 配置控制。
- **Risk:** data 模块引入 W&B 后影响本地路径读取。→ **Mitigation:** data 读取入口只在 `wandb://` URI 出现时延迟导入 W&B backend；普通本地路径继续走现有 `open_local_or_remote` / `torch.load` 流程，不要求 W&B 凭据。
- **Risk:** lineage callback 依赖 callback 配置，用户可能只写 `wandb://` 但未启用 callback。→ **Mitigation:** 官方 W&B 实验配置可以显式包含 `WandbArtifactLineageCallback`；非 W&B/本地配置保持无副作用。错误或 warning 应说明“已解析但未记录 lineage”的原因。

## Migration Plan

1. 新增 `src/data/components/artifacts.py`，迁移 `load_model_output` / `load_semantic_id_tensor`，实现字段级 W&B reference resolver 和 resolved-reference registry；新增 `src/utils/wandb.py` 承载底层 W&B URI、Artifact 选择和下载 helper，并添加单元测试。
2. 将 writer 能力整理到 `src/common/writers/`，删除旧 `src/common/inference/`，并将 `LocalPickleWriter`、`WandbArtifactWriter`、checkpoint writer 拆分为独立文件；W&B writer 只发布已存在 bundle 文件。
3. 新增 checkpoint writer，基于 `ModelCheckpoint` 已写出的 best/last/explicit checkpoint 发布，不改变 `ModelCheckpoint` 与模型配置职责。
4. 在 `ckpt_path`、`embedding_path`、`semantic_id_path` 的局部消费边界接入通用 resolver；不得在 launcher 中做 W&B 配置扫描。
5. 在 `src/common/callbacks/` 新增显式 `WandbArtifactLineageCallback`，用于把 registry 中的上游 Artifact 记录到当前 active W&B run。
6. 更新官方脚本示例和错误提示，说明本地路径与 `wandb://<run-id>` 均可使用。
7. 用 Hydra compose、单元测试和 dry-run 验证本地路径流程不会触发 W&B。

## Open Questions

无阻塞未决问题。实现时仍需按当前 W&B SDK 实际对象 API 确认 output Artifact 枚举字段名称，但这属于实现细节，不影响方案边界。
