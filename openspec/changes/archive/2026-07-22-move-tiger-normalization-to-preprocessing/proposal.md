## Why

`collate_fn_train` 曾在对 TIGER 训练输入执行 `normalize_sequence_batch` 并生成 `attention_mask`，推理侧 `collate_fn_inference_for_sequence` 也承担同类长度规范化职责。将长度规范化和 attention mask 生成前移到 dataset preprocessing，可以让统一的 `collate_fn_sequence` 收敛为真正的 batch assembly，并让 TIGER train/eval/inference 的数据准备顺序在配置中完整可见。

## What Changes

- 新增 row-level preprocessing 函数 `normalize_sequence`，用于将单条 row 的 `input_ids` pad/trim 到固定 `sequence_length`，并基于规范化后的 `input_ids` 生成 `attention_mask`。
- 在 `configs/data/tiger_train.yaml` 的 train/eval preprocessing chain 中，将 `normalize_sequence` 放在 `generate_next_k_labels` 之后；在 `configs/data/tiger_inference.yaml` 的 semantic ID 映射之后声明 `normalize_sequence`。
- 将 `collate_fn_train` 重命名并泛化为 `collate_fn_sequence`：直接 stack 预处理后的 input、attention mask、可选 target 和可选 output keys，不再调用 `normalize_sequence_batch`，不再接收 `sequence_length` 或自行生成 `attention_mask`。
- 保持 `generate_next_k_labels` 在 normalization 之前运行，确保 `target_ids` 仍来自真实或上采样后的 semantic-ID 序列末尾。
- 将 `collate_fn_inference_for_sequence` 的用法替换为 `collate_fn_sequence`，推理 output keys 通过 `output_key_field_name` 进入 `TigerModelInput.output_keys`。
- **BREAKING**：TIGER sequence rows entering `collate_fn_sequence` must already contain fixed-length input and `attention_mask` fields.

## Capabilities

### New Capabilities

- `preprocessed-sequence-normalization`: 定义 row-level `normalize_sequence` preprocessing helper 的输入、输出、顺序和 attention mask 生成协议。

### Modified Capabilities

- `config-declared-preprocessing-contract`: TIGER preprocessing chain 必须显式声明 label generation 后的 sequence normalization step。
- `tiger-preprocessed-label-generation`: label generation 后必须继续执行 input normalization，collate 前 rows 必须包含 fixed-length `input_ids` 与 `attention_mask`。
- `unified-tiger-train-collate`: `collate_fn_sequence` 的职责收敛为只 stack 已预处理字段并构造 dataclass。
- `tiger-specific-batch-contract`: TIGER train/eval/inference batch 的 `input_ids` 和 `attention_mask` 均来自预处理字段，`target_ids` 来自预生成标签字段，推理 output keys 来自配置字段。
- `tiger-sequence-data-contract`: TIGER sequence 的 `sequence_length` 从 collate 配置迁移到 preprocessing 配置。

## Impact

- Code:
  - `src/data/components/preprocessing.py`
  - `src/data/components/collate.py`
  - optionally `src/data/utils.py` if single-row normalization logic is extracted for reuse
- Config:
  - `configs/data/tiger_train.yaml`
- Specs:
  - 新增 `preprocessed-sequence-normalization`
  - 更新上述 TIGER preprocessing / collate / data contract living specs
- Verification focus:
  - `normalize_sequence` row-level smoke for pad, trim, and attention mask
  - `collate_fn_sequence` smoke confirming it only stacks preprocessed fields and preserves inference output keys
  - `tiger_train` / `tiger_inference` Hydra compose + datamodule instantiate smoke
  - compile/ruff for touched Python files
  - OpenSpec strict validation and full specs validation
