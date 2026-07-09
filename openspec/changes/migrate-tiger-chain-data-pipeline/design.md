## Context

tiger_train 和 tiger_inference 是 GRID pipeline 最后两段——sequence 链路的实验。它们消费上游 rkmeans_inference 产生的 semantic ID keyed bundle，将用户交互序列映射为 semantic ID 序列，训练/推理 TIGER 生成式推荐模型。

当前这两个实验的 data 配置停留在旧架构：`data_reader` 直接实例化、preprocessing 分散命名引用 + resolver 派生、`map_sparse_id_to_semantic_id` 接收整个 `dataset_config` 从中读 `semantic_id_map`、shuffle 经 `should_shuffle_rows` 控制。`SequenceDatasetConfig` 携带 7 个无运行时消费方的遗留字段，`SequenceDataloaderConfig` 残留 `should_shuffle_rows`。

新 data contract 的基础设施已就绪：`map_sparse_id_to_embedding` 已完成局部 bundle 参数迁移（可作为参照）、`BaseDataModule` 已从 `dataset_config.shuffle_files` 读文件级 shuffle、`SequenceDataModule._build_collate_fn` 已正确绑定 sequence 专属参数。item 链路 5 个实验已全部迁移完成。

## Goals / Non-Goals

**Goals:**
- 将 tiger_train / tiger_inference 两个 data 配置迁移到新 data contract。
- 重构 `map_sparse_id_to_semantic_id` 为接收局部 `semantic_id_bundle` 参数，消除最后一个违反 preprocessing-minimal-parameter-contract 的 preprocessing 函数。
- 扁平化 `SequenceDatasetConfig` 到 `SemanticIDDatasetConfig`，清理全部遗留字段。
- 从 `SequenceDataloaderConfig` 删除 `should_shuffle_rows`，统一 shuffle 语义。
- 清理 collate 配置中的冗余参数（被 `_build_collate_fn` 覆盖的 `sequence_length`/`padding_token`）。

**Non-Goals:**
- 不改动 `SequenceDataModule._build_collate_fn`（已有正确逻辑）。
- 不改动 collate 函数签名（`collate_with_sid_causal_duplicate`/`collate_fn_train`/`collate_fn_inference_for_sequence`）。
- 不改动 `BaseDataModule`、`SequenceDataset`、`TFRecordReader`。
- 不改动 experiment 配置（`configs/experiment/tiger_train.yaml`、`tiger_inference.yaml`）。
- 不删除 Hydra resolver（迁移后可考虑在后续变更中统一清理）。
- 不迁移 `file_format` 字段（tiger 配置未显式设置，`BaseDataModule` 已有 `data_reader.get_file_suffix()` fallback）。

## Decisions

### 决策 1：`map_sparse_id_to_semantic_id` 重构为局部 bundle 参数

**选择**：将 `dataset_config` 参数替换为 `semantic_id_bundle` 局部参数，完全类比 `map_sparse_id_to_embedding`。

**理由**：`map_sparse_id_to_embedding` 已完成相同迁移，签名模式经验证可用。旧函数从 `dataset_config.semantic_id_map.get(k)` 取 bundle，但 bundle 本质上是 preprocessing 步骤的输入数据，应作为局部参数注入而非挂在 dataset config 上。

**备选**：保留 `dataset_config` 参数但只从中取 `semantic_id_map`。放弃——违反 preprocessing-minimal-parameter-contract，且与 `map_sparse_id_to_embedding` 不一致。

### 决策 2：扁平化 `SequenceDatasetConfig` 到 `SemanticIDDatasetConfig`

**选择**：删除 `SequenceDatasetConfig` 类，`SemanticIDDatasetConfig` 成为独立类（不继承），仅保留 3 个运行时消费字段：`data_reader`、`preprocessing_functions`、`shuffle_files`。

**理由**：`SequenceDatasetConfig` 从未被直接实例化（无 YAML 以它为 `_target_`），唯一子类 `SemanticIDDatasetConfig` 仅在 tiger 链路使用。11 个字段中仅 3 个运行时消费（`file_format` 也删除，因 tiger 配置未设置且 `BaseDataModule` 有 fallback）。保留空壳基类无意义。

**备选**：保留继承但清理字段。放弃——增加无谓复杂度，两阶段重构不如一次到位。

### 决策 3：`user_id_field` 从 dataset_config 删除，collate 硬编码

**选择**：从 `SemanticIDDatasetConfig` 删除 `user_id_field`，tiger_inference 的 collate 硬编码 `id_field_name: user_id`。

**理由**：`user_id_field` 在运行时无 Python 代码从 dataset_config 实例读取（`SequenceDataset`/`BaseDataModule`/preprocessing 均不读）。唯一消费方是 `collate_fn_inference_for_sequence` 的 `id_field_name` 参数，通过 Hydra 插值引用。与 item 链路 `item_id_field` 的处理完全一致——collate 硬编码，dataset_config 不携带。

**备选**：保留 `user_id_field` 在 dataset_config 上作为 collate 插值源。放弃——与 item 链路不一致，且字段无运行时消费方。

### 决策 4：sequence dataloader config 保留 sequence 专属字段

**选择**：`SequenceDataloaderConfig` 保留 `labels`/`masking_token`/`sequence_length`/`padding_token`/`oov_token`，仅删除 `should_shuffle_rows`。

**理由**：`SequenceDataModule._build_collate_fn` 从 dataloader config 读取这 5 个字段绑定到 collate 函数。这是 sequence 链路本质需求——collate 需要知道序列长度、padding token、label 函数等。与新 item 链路（`ItemDataModule._build_collate_fn` 直接返回 collate）不同，不能照搬 item 模板删字段。

**备选**：将 sequence 专属字段移到 collate config。放弃——`_build_collate_fn` 从 dataloader config 统一绑定，拆散会破坏现有逻辑。

### 决策 5：清理 collate 配置冗余参数

**选择**：`train_collate` 删除 `sequence_length` 和 `padding_token`（被 `_build_collate_fn` 覆盖），保留函数专属参数（`sequence_field_name`、`sid_hierarchy`、`max_batch_size`）。`eval_collate` 同样删除冗余参数。

**理由**：`functools.partial` 嵌套绑定时，外层（`_build_collate_fn`）的 kwargs 覆盖内层（collate _partial_）的值。`train_collate` 的 `sequence_length`/`padding_token` 永远被覆盖，是死参数。删除后配置更清晰。

**备选**：保留冗余参数。放弃——误导维护者以为参数生效，且与清理主题不符。

### 决策 6：`features_to_consider` 显式声明为 `[sequence_data, user_id]`

**选择**：preprocessing_functions 的 `filter_features_to_consider` 显式声明 `features_to_consider: [sequence_data, user_id]`，不通过 resolver 从 `features` list 派生。

**理由**：旧配置通过 `extract_fields_from_list_of_dicts` resolver 从 `features` list 派生 `features_to_consider`，实际只需保留 `sequence_data`（SID 序列）和 `user_id`（推理 key 映射）。显式声明符合 config-declared-preprocessing-contract，且消除 resolver 依赖。tiger_train 的 `user_id` 需保留（`collate_with_sid_causal_duplicate` 和 `collate_fn_train` 将其作为序列特征处理）；tiger_inference 的 `user_id` 需保留（`collate_fn_inference_for_sequence` 用 `id_field_name` 做 key 映射）。

## Risks / Trade-offs

- **[collate 冗余参数删除可能影响 `partial` 解析]** 删除 `train_collate` 的 `sequence_length`/`padding_token` 后，`collate_with_sid_causal_duplicate` 的这些参数完全由 `_build_collate_fn` 绑定。→ **缓解**：dry-run smoke check 验证 collate 实例化 + 数据加载正常。
- **[`SequenceDatasetConfig` 删除影响类型注解]** `SequenceDataloaderConfig.dataset_config` 类型注解从 `SequenceDatasetConfig` 改为 `SemanticIDDatasetConfig`。→ **缓解**：类型注解仅为文档目的，Hydra 实例化不检查类型，改动安全。
- **[迁移前配置可能已坏]** tiger 配置是否当前可实例化未验证（sequence 链路 config 类字段可能已被先前变更影响）。→ **缓解**：迁移后以新 contract 为准做 smoke check，不依赖迁移前 baseline。
- **[`map_sparse_id_to_semantic_id` 签名变化是 BREAKING]** 任何其他调用方都会受影响。→ **缓解**：grep 确认唯一调用方是 tiger 配置的 preprocessing_functions；无其他 Python 代码直接调用。
