## Why

GRID 的实验链路目前依赖长本地路径在阶段之间传递 checkpoint、embedding、semantic ID 和 inference output，用户在启动下一阶段时需要复制易错且不便比较的路径。W&B 已经作为训练实验记录入口使用，但当前产物没有形成可查询的 Artifact lineage，导致多个实验之间的输入输出关系无法在 W&B 中直观看到。

## What Changes

- 为阶段间文件引用增加兼容式 W&B run 引用协议，允许现有 `ckpt_path`、`embedding_path`、`semantic_id_path` 等字段继续接受本地路径，同时支持 `wandb://<run-id>`。
- 新增 W&B Artifact lineage 能力：训练与推理阶段可选地将 checkpoint 或 model output bundle 记录为 W&B Artifact，并在消费上游产物时通过 `use_artifact` 建立 lineage。
- 删除旧 `src/common/inference/`，将 writer 能力整理到 `src/common/writers/`，让 `LocalPickleWriter`、`WandbArtifactWriter` 与 checkpoint writer 平级且分别位于独立文件。
- 将 `wandb://...` 字段级引用解析归入 data artifact 读取职责：在 `src/data/components/artifacts.py` 中承载 `load_model_output`、`load_semantic_id_tensor`、字段默认 role 和 resolved-reference registry；底层 W&B URI parser、Artifact selector/downloader 等工具函数放在 `src/utils/wandb.py`；`ckpt_path` 在 checkpoint 读取边界复用该 resolver。
- 新增 `WandbArtifactWriter`，用于把已经落盘的 model output bundle 发布为 Artifact；训练 checkpoint 使用独立 checkpoint writer。
- 新增 `WandbArtifactLineageCallback`，位于 `src/common/callbacks/`，只负责读取 data artifact registry 并在 active W&B run 中调用 `use_artifact`。
- 保持 logger、local writer、W&B writer 低耦合：使用 W&B logger 不要求使用 W&B writer；使用本地 `LocalPickleWriter` 不要求启用 W&B；W&B writer 只由自身配置显式启用。
- 脚本参数保持向后兼容：当前本地路径参数继续有效，仅扩展参数值协议和可选 W&B 配置。
- 不引入破坏性变更。

## Capabilities

### New Capabilities

- `wandb-artifact-lineage`: 定义可选 W&B Artifact 输出、`wandb://` run 引用解析、Artifact 下载缓存、当前 run 使用上游 Artifact 的 lineage 记录，以及模块低耦合边界。

### Modified Capabilities

- `keyed-prediction-bundle-artifact`: keyed prediction bundle 的路径引用可由本地路径扩展为 W&B run 引用，但 bundle 文件内容协议不变。
- `prediction-output-protocol`: 推理本地 writer 继续只负责本地 batch output 合并；Artifact 发布必须作为独立可选能力接入。
- `model-training-components`: checkpoint 可选发布为 W&B Artifact，并可通过 `ckpt_path=wandb://<run-id>` 解析；训练模型配置职责不变。

## Impact

- 影响 `src/common/writers/`：承载本地与 W&B writer，`LocalPickleWriter`、`WandbArtifactWriter` 与 checkpoint writer 平级且按文件拆分；旧 `src/common/inference/` 不保留兼容导入。
- 影响 `src/common/callbacks/`：新增 `WandbArtifactLineageCallback`，作为生命周期 callback 记录上游 Artifact lineage，不作为 logger 或 writer。
- 影响 `src/data/components/artifacts.py`：新增或迁移 `load_model_output` / `load_semantic_id_tensor`，并承载字段默认 role、引用解析入口和 resolved-reference registry。
- 影响 `src/utils/wandb.py`：承载 W&B URI parser、Artifact selector/downloader 和下载文件定位等基础工具函数。
- 影响 checkpoint 读取/写入边界：新增 checkpoint URI resolver 与 checkpoint writer，用于下载或发布 checkpoint；默认本地 checkpoint 行为保持。
- 影响 path 消费边界：`ckpt_path`、`embedding_path`、`semantic_id_path` 在各自使用前解析；launcher 不做 W&B 配置判断。
- 影响 shell 脚本与配置：参数名保持不变，文档和错误提示补充 `wandb://<run-id>` 用法。
- 需要 W&B SDK；项目当前已依赖 `wandb>=0.28.0`，不预期新增核心依赖。
