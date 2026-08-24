## 1. W&B Reference 基础模块

- [x] 1.1 新增 `src/data/components/artifacts.py`，迁移或承载 `load_model_output`、`load_semantic_id_tensor` 等 artifact/bundle 加载函数。
- [x] 1.2 在 `src/utils/wandb.py` 中新增独立 W&B URI parser，支持 `wandb://<run-id>`、`wandb://<entity>/<project>/<run-id>` 和 `role`/`alias`/`file` 查询参数。
- [x] 1.3 新增 reference resolver，按字段名推断默认 role，并对非 `wandb://` 输入保持零副作用透传。
- [x] 1.4 在 `src/utils/wandb.py` 实现 deterministic Artifact selection：0 个或多个匹配时给出明确错误，不静默猜测。
- [x] 1.5 在 `src/utils/wandb.py` 实现 Artifact 下载缓存并返回具体本地文件路径。
- [x] 1.6 新增 resolved-reference registry，记录原始引用、producer run、artifact version、resolved local path。
- [x] 1.7 为 parser、resolver、错误路径、本地路径 bypass、bundle 加载和 registry 添加单元测试。

## 2. Lineage 与 Artifact Writer

- [x] 2.1 在 `src/common/callbacks/` 实现显式配置的 `WandbArtifactLineageCallback`：只在 active W&B run 存在时对 registry 中的输入调用 `use_artifact`。
- [x] 2.2 将 writer 能力整理到 `src/common/writers/`，删除旧 `src/common/inference/` 兼容导入，并将各 writer 拆分为独立模块。
- [x] 2.3 在 `src/common/writers/` 新增与 `LocalPickleWriter` 平级的 `WandbArtifactWriter`，支持 role、metadata、file name、alias 配置。
- [x] 2.4 新增 checkpoint writer，用于发布 `ModelCheckpoint` 已写出的 best/last/explicit checkpoint，不改变 `ModelCheckpoint` 行为。
- [x] 2.5 为无 active W&B run、writer 关闭、发布失败策略添加测试。
- [x] 2.6 确保 W&B writer 不依赖 W&B logger、不要求搭配 `LocalPickleWriter`，也不改变本地 writer 行为。
- [x] 2.7 为 checkpoint writer 添加测试：唯一 `ModelCheckpoint` 发布 best path、无 checkpoint skip/fail、多 checkpoint 未指定时报错、metadata 记录 monitor/mode/best score。

## 3. Path 消费边界接入

- [x] 3.1 在 `embedding_path`、`semantic_id_path` 的 data 读取边界解析 `wandb://` 引用，尤其覆盖 `load_model_output` / `load_semantic_id_tensor`。
- [x] 3.2 确保 `ckpt_path=wandb://<run-id>` 在 Trainer 使用 checkpoint 前解析为本地 checkpoint 文件。
- [x] 3.3 确保 `embedding_path`、`semantic_id_path` 经 resolver 后仍由 `load_model_output` 和 `load_semantic_id_tensor` 按现有 bundle 协议读取。
- [x] 3.4 确保 launcher 不扫描 Hydra 配置、不判断 W&B 引用、不提前实例化 logger、不直接调用 W&B resolver/writer。
- [x] 3.5 添加本地路径 compose/dry-run 测试，证明未使用 W&B 协议时不初始化 W&B。

## 4. 训练与推理产物配置

- [x] 4.1 为训练 checkpoint Artifact 发布增加可选配置，默认不改变本地 checkpoint 行为。
- [x] 4.2 为 inference `merged_predictions_tensor.pt` 增加可选 `WandbArtifactWriter` 配置，默认不改变 `LocalPickleWriter` 行为。
- [x] 4.3 在 Artifact metadata 中记录 role、task_name、local path、bundle file、producer run 信息。
- [x] 4.4 覆盖 `semantic_embedding`、`semantic_id`、`recommendation_output`、`checkpoint` 四类 role。

## 5. 脚本、文档与验证

- [x] 5.1 更新官方脚本说明和错误提示，保留当前参数名并展示 `wandb://<run-id>` 示例。
- [x] 5.2 更新相关 living specs 或 AGENTS.md 规范，强调 W&B 模块可选、低耦合、非本地路径默认行为。
- [x] 5.3 运行针对 resolver、registry、writer、`WandbArtifactLineageCallback` 和 path 消费边界的 focused pytest。
- [x] 5.4 运行相关 Hydra compose 验证本地路径和 `wandb://` 配置形态。
- [x] 5.5 运行 `openspec validate add-wandb-artifact-lineage --strict`。
