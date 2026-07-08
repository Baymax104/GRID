## Context

三个 item 级量化实验（rvq_train、rqvae_train、rkmeans_inference）的 data 配置仍停留在旧架构：`data_reader` 直接实例化、preprocessing 通过分散命名引用 + Hydra resolver 派生拼装、embedding 经 `dataset.embedding_map` 注入、shuffle 经 dataloader 的 `should_shuffle_rows` 控制。由于 `remove-legacy-item-data-config-fields` 已从 `ItemDatasetConfig` / `ItemDataloaderConfig` 删除旧字段，这三个配置当前已无法实例化。

新 data contract 的全部基础设施已就绪：`map_sparse_id_to_embedding` 已改为接收局部 `embedding_bundle`、`collate_fn_items` 签名不变、`TFRecordReader` 支持 factory 化、`BaseDataModule` 已从 `dataset_config.shuffle_files` 读文件级 shuffle。rkmeans_train 已完成迁移并作为可参照样板。

## Goals / Non-Goals

**Goals:**
- 将 rvq_train / rqvae_train / rkmeans_inference 三个 data 配置迁移到与 rkmeans_train 一致的新架构。
- 消除坏配置，使这三个实验恢复可实例化。
- 统一 item 链路 shuffle 语义（shuffle_files + shuffle_rows），彻底脱离 should_shuffle_rows。

**Non-Goals:**
- 不改动任何 Python 代码（preprocessing 函数、config 模型、collate、datamodule 均不动）。
- 不迁移 tiger_train / tiger_inference sequence 链路（需单独重构 `map_sparse_id_to_semantic_id`，另开变更）。
- 不删除 Hydra resolver `extract_fields_from_list_of_dicts` / `create_map_from_list_of_dicts`（sequence 链路仍引用）。
- 不改动 experiment 配置（`configs/experiment/*.yaml`）的组装结构。

## Decisions

### 决策 1：照搬 rkmeans_train 样板，不做架构变体

**选择**：三个实验的 data 配置结构完全对齐 rkmeans_train，仅按 train/val/test vs predict-only 区分 dataloader 数量。

**理由**：rkmeans_train 是同类型 item 量化实验且已验证可用，其 preprocessing chain（filter_features_to_consider → convert_to_dense_numpy_array → convert_fields_to_tensors → map_sparse_id_to_embedding）与这三个实验的业务逻辑一致——都是按 item_id 读取、注入 embedding。照搬可消除结构性分歧，降低维护成本。

**备选**：保留各实验原有 preprocessing 步骤差异。放弃——三者原有步骤经核查与 rkmeans_train 完全等价，无实质差异需保留。

### 决策 2：shuffle 语义映射规则

**选择**：旧 `should_shuffle_rows: true` → `shuffle_files: true` + `shuffle_rows: true`；旧 `should_shuffle_rows: false` → `shuffle_files: false` + `shuffle_rows: false`。

**理由**：旧 `should_shuffle_rows` 实际承载了文件级与样本级两层 shuffle 语义（旧 reader 单一开关同时控制两者）。拆分时保持等价行为最安全。对 train 阶段两者均开，对 eval/predict 阶段两者均关，与 rkmeans_train 样板一致。

**备选**：train 阶段 `shuffle_files: false + shuffle_rows: true`（只打乱样本不打乱文件）。放弃——与 rkmeans_train 样板不一致，且文件级 shuffle 对多 worker 负载均衡有益。

### 决策 3：rkmeans_inference 的 feature_to_input_name 移到顶层

**选择**：将 rkmeans_inference 原内嵌在 `predict_dataloader` 内的 `feature_to_input_name` 提升到 data 配置顶层，collate 引用改为 `${data.feature_to_input_name}`。

**理由**：rkmeans_train 样板中 `feature_to_input_name` 在顶层，被所有 dataloader 的 collate 共享。rkmeans_inference 虽只有 predict_dataloader，但对齐顶层放置可保持 item 链路一致，且减少嵌套深度。collate 的 `item_id_field` 引用同步改为 `${data.predict_dataset_config.item_id_field}`。

**备选**：保留 feature_to_input_name 内嵌在 predict_dataloader。放弃——与样板不一致，后续若新增 stage 需重复定义。

### 决策 4：rvq_train 与 rqvae_train 配置完全一致

**选择**：两个文件迁移后保持逐字一致（与迁移前一致）。

**理由**：迁移前二者已逐字相同（111 行），它们仅在 model 层面有差异（RVQ vs RQ-VAE），data 层完全共享。迁移后维持一致避免无谓分歧。

## Risks / Trade-offs

- **[resolver 残留引用]** item 链路迁移后不再引用 `extract_fields_from_list_of_dicts` / `create_map_from_list_of_dicts`，但 resolver 定义保留。→ **缓解**：迁移后 grep 确认 item 链路无残留引用；resolver 待 sequence 链路迁移完毕后另开变更统一清理。
- **[迁移前配置已坏]** 三个配置当前无法实例化，无法做"迁移前 baseline"对比。→ **缓解**：迁移后以 rkmeans_train 样板为参照做 dry-run smoke check 验证可实例化与 row 产出。
- **[rvq_train / rqvae_train 无独立运行脚本]** 仓库根无 rvq_train.sh / rqvae_train.sh，仅 experiment 配置存在。→ **缓解**：smoke check 通过 `uv run python -m src.main experiment=rvq_train --dry-run ...` 直接验证，无需脚本。
