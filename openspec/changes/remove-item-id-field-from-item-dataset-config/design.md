## Context

`ItemDatasetConfig` 当前有 4 个字段：`item_id_field`、`data_reader`、`preprocessing_functions`、`shuffle_files`。其中 `data_reader`/`preprocessing_functions`/`shuffle_files` 由 `SequenceDataset.__init__` 和 `BaseDataModule.setup` 运行时消费，而 `item_id_field` 仅被 `collate_fn_items` 通过 collate `_partial_` 配置消费——无任何 Python 代码从 `dataset_config` 实例上读取它。

此前 rvq_train/rqvae_train/rkmeans_inference 的 collate 通过 Hydra 插值 `${data.*_dataset_config.item_id_field}` 引用该字段，rkmeans_train/sem_embeds_inference 则直接硬编码 `item_id_field: id`。用户已将 rvq_train/rqvae_train 的插值改为硬编码，仅 rkmeans_inference 残留一处插值。

## Goals / Non-Goals

**Goals:**
- 从 `ItemDatasetConfig` 移除 `item_id_field`，使该 config 类仅保留 dataset 运行时消费的字段。
- 统一所有 5 个 item 链路配置的 collate 块为硬编码 `item_id_field: id`。
- 消除 rkmeans_inference.yaml 残留的 `${data.predict_dataset_config.item_id_field}` 插值引用。

**Non-Goals:**
- 不改动 `collate_fn_items` 函数签名或逻辑。
- 不处理 sequence 链路的 `user_id_field`（`SequenceDatasetConfig` 上同理存在，但属于 sequence 链路迁移范围）。
- 不改动 `feature_to_input_name` 的归属（它已在 `ItemDataloaderConfig` 上定义，collate 通过顶层 `${data.feature_to_input_name}` 引用）。

## Decisions

### 决策 1：collate 硬编码 `item_id_field: id`，不引入新插值

**选择**：所有 5 个配置的 collate 块统一硬编码 `item_id_field: id`。

**理由**：所有 item 链路实验的 item ID 字段名固定为 `id`，无变体需求。硬编码消除了对 dataset config 的跨域引用，配置更直观。引入新插值（如 `${data.collate.item_id_field}`）只是把冗余搬到别处，无收益。

**备选**：将 `item_id_field` 移到 `ItemDataloaderConfig` 并插值引用。放弃——增加了 config 类字段但无运行时收益，且所有实验值相同无参数化需求。

### 决策 2：不改动 `collate_fn_items` 签名

**选择**：`collate_fn_items(batch, item_id_field, feature_to_input_name)` 签名保持不变。

**理由**：函数本身正确地需要 `item_id_field` 来区分 ID 与特征。问题在于该值的配置归属（dataset config vs collate config），而非函数接口。改签名会波及所有 collate 调用方，得不偿失。

## Risks / Trade-offs

- **[硬编码缺乏灵活性]** 若未来某实验的 item ID 字段名不是 `id`，需修改 collate 配置。→ **缓解**：当前所有实验均用 `id`，且 collate 配置是 per-experiment 的 YAML，修改成本低。
- **[与 sequence 链路不对称]** `SequenceDatasetConfig` 仍有 `user_id_field`，item 链路却不再有 `item_id_field`。→ **缓解**：这是预期的——sequence 链路尚未迁移，待其迁移时统一处理。两链路的 collate 函数不同（`collate_fn_items` vs `collate_fn_train`），配置模式可以独立演化。
