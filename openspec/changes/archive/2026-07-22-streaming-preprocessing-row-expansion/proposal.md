## Why

TIGER 的 label generation 和 SID causal duplicate augmentation 仍然位于 `collate_fn_train`，使 collate 同时承担样本扩展、label 生成和 batch 拼装职责。将 dataset preprocessing 扩展为流式 row expansion 可以把这些训练样本构造逻辑前移，同时避免 list pipeline 在大数据集上物化和重复遍历扩展结果。

## What Changes

- 扩展 `SequenceDataset` preprocessing contract：preprocessing function 可以返回单条 row、`None`，或一个可迭代的多条 rows；dataset 以 streaming flat-map 方式继续执行后续 preprocessing 并逐条 yield。
- 新增 row-level SID causal duplicate expansion preprocessing helper，用 generator 产出 semantic-ID-aligned contiguous subsequences，不在 collate 中做 batch-level augmentation。
- 新增 row-level TIGER next-item label generation preprocessing helper，将 `sequence_data` 转为 `input_ids` 和 `target_ids`。
- 简化 `collate_fn_train`：只读取预处理后的 `input_ids` / `target_ids`，执行 padding/stack/attention mask 和 TIGER data container 拼装。
- 更新 `configs/data/tiger_train.yaml`，将训练 augmentation 和 train/eval label generation 移入 dataset preprocessing chain，并移除 collate 中的 label/augmentation 参数。
- **BREAKING**: `collate_fn_train` 不再接收 `label_generate_functions`、`masking_token`、`enable_sid_causal_duplicate`、`sequence_field_name`、`sid_hierarchy`、`max_batch_size` 等 label/augmentation 参数。

## Capabilities

### New Capabilities
- `streaming-preprocessing-row-expansion`: Defines streaming flat-map preprocessing where a preprocessing step can filter, keep, or expand rows without materializing the full expanded list.
- `tiger-preprocessed-label-generation`: Defines TIGER row-level preprocessing that generates `input_ids` and `target_ids` before collate.

### Modified Capabilities
- `config-declared-preprocessing-contract`: Preprocessing configuration must be able to declare row expansion and label generation steps explicitly.
- `tiger-sequence-data-contract`: TIGER sequence train/eval preprocessing and collate responsibilities move label generation out of collate.
- `unified-tiger-train-collate`: The unified collate entry point remains, but its responsibility narrows to pure batch assembly.
- `pure-label-function-contract`: Label generation remains function-based but becomes preprocessing-owned instead of collate-owned.
- `tiger-specific-batch-contract`: TIGER batch containers are assembled from precomputed `input_ids` and `target_ids` fields.

## Impact

- Affected code:
  - `src/data/datasets.py`
  - `src/data/components/preprocessing.py`
  - `src/data/components/label_functions.py`
  - `src/data/components/collate.py`
  - `configs/data/tiger_train.yaml`
- Affected OpenSpec specs:
  - New specs listed above
  - Existing TIGER preprocessing/collate/data contract specs listed above
- Validation focus:
  - streaming expansion smoke for `SequenceDataset`
  - SID causal duplicate expansion smoke without materializing full batch expansion
  - TIGER train/eval collate smoke using preprocessed `input_ids` / `target_ids`
  - Hydra compose/instantiate for `tiger_train`
  - `openspec validate --specs --no-interactive`
