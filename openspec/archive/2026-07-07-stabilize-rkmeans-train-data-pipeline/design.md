## Context

本仓库的 data 重构正在沿以下方向收敛：

- `data_iterator` → `data_reader`
- dataset 不再长期持有可变 reader instance，而是在运行时按需构造 reader
- preprocessing function 改为通过 Hydra `_partial_` 预绑定 config，运行时只接收 `row_or_batch`
- shuffle 语义拆为文件层和样本层，而不是混在一个 `should_shuffle_rows` 开关里

`rkmeans_train` 已经开始使用这套新思路：

- `dataset_config.data_reader` 配成 `_partial_` 的 `TFRecordReader`
- `preprocessing_functions` 采用预绑定 `dataset_config` 的列表形式
- `dataset_config.shuffle_files` 与 `data_reader.shuffle_rows` 已出现在配置中

但链路尚未稳定，仍有旧接口残留，例如：

- `BaseDataModule._build_dataset()` 仍按旧参数名传 `global_worker_id`
- `ItemDatasetConfig` 仍把 `data_reader` 写成实例型字段，而非 factory contract
- `should_shuffle_rows` 仍残留在 dataloader config 与 file assignment 路径中
- `rkmeans_train.yaml` 中仍存在 `${data.dataset}` 等旧引用

## Goals / Non-Goals

**Goals:**
- 让 `rkmeans_train` 成为第一个稳定的新 data pipeline 模板
- 明确 `data_reader` 的最终 contract 为 factory / partial
- 明确 shuffle contract 只包含 `shuffle_files` 与 `shuffle_rows`
- 清理 `rkmeans_train` 上与旧 data 链路混杂的残留接口

**Non-Goals:**
- 不要求本次把所有 experiments 同时迁到新模式
- 不改变 `rkmeans_train` 的业务语义、embedding lookup 语义或 collate 输出结构
- 不在本次重构所有 preprocessing 函数实现细节，除非为匹配新 contract 所必需

## Decisions

### D1: `data_reader` 的最终 contract 采用 factory / partial 模式
- **选择**：`SequenceDatasetConfig` / `ItemDatasetConfig` 中的 `data_reader` 统一表示为可调用 factory（通常是 Hydra `_partial_`）
- **理由**：这与当前 `SequenceDataset._load_data()` 的新方向一致，也避免 dataset 长期持有可变 reader instance
- **备选**：继续接受 instance 模式 —— 否决，会把 mutation 驱动的旧设计继续带入新链路

### D2: `SequenceDataset` 在 `_load_data()` 中按需实例化 reader
- **选择**：dataset 只持有 reader factory，不持有长期复用的 reader instance
- **理由**：这样每轮迭代都可基于当前 worker 文件列表显式构造新的 reader，状态边界清晰

### D3: shuffle contract 收敛为 `shuffle_files` + `shuffle_rows`
- **选择**：
  - `dataset_config.shuffle_files`：控制 worker 内文件顺序
  - `data_reader.shuffle_rows`：控制 reader 内样本行 shuffle
- **理由**：这是当前用户明确选择的最终方向
- **备选**：保留 `should_shuffle_rows` 作为长期入口 —— 否决，仅允许作为迁移期清理对象，不再作为目标 contract

### D4: `BaseDataModule` 不再依赖 dataloader config 的 `should_shuffle_rows`
- **选择**：file assignment / dataset construction 路径改为读取 dataset_config 中的新 shuffle 字段，而非 dataloader config 的旧字段
- **理由**：shuffle 语义属于 dataset / reader contract，而不是 dataloader transport contract

### D5: 本次仅稳定 `rkmeans_train`，但同时收紧其依赖的最小公共接口
- **选择**：只修改会直接影响 `rkmeans_train` 稳定性的 data 核心模块与配置
- **理由**：既控制改动面，又为后续迁移其他实验提供稳定模板

## Risks / Trade-offs

- **[风险] 过早移除 `should_shuffle_rows` 可能影响尚未迁移的其他 experiments**  
  **缓解**：本次提案明确以 `rkmeans_train` 为主，必要时仅在不影响其稳定性的前提下保留其他实验的旧字段兼容，但不把兼容写入新 contract。

- **[风险] `data_reader.get_file_suffix()` 目前仍被 datamodule 直接调用，若 `data_reader` 是 factory，接口处理方式需明确**  
  **缓解**：本次将其纳入 `data-reader-factory-contract` 的实现范围，统一 reader suffix 的获取方式。

- **[权衡] 本次不会同时修完所有实验**  
  **可接受**：目标是先把 `rkmeans_train` 跑通并定型，再批量迁移其他实验。

## Migration Plan

1. 统一 `config_models.py` 中 `data_reader` 的 factory 类型契约，并补齐 `shuffle_files`
2. 对齐 `BaseDataModule` 与 `SequenceDataset` 的新构造接口
3. 从 `rkmeans_train.yaml` 中移除旧的 `data.dataset` / `should_shuffle_rows` 残留
4. 统一 file suffix 与 shuffle 路径对新 contract 的读取方式
5. 对 `rkmeans_train` 做最小 compose / import / 实例化验证，确认其可作为后续迁移模板
