## 1. Row-level normalization preprocessing

- [x] 1.1 Add `normalize_sequence` to `src/data/components/preprocessing.py` with configurable `input_field_name`, `attention_mask_field_name`, `sequence_length`, and `padding_token`
- [x] 1.2 Ensure `normalize_sequence` pads short 1-D tensors, trims long 1-D tensors using existing TIGER normalization semantics, and rejects non-1-D input tensors with a clear error
- [x] 1.3 Ensure `normalize_sequence` writes `attention_mask` from the normalized input field and does not mutate `target_ids`
- [x] 1.4 Reuse or extract shared single-sequence normalization logic from `src/data/utils.py` if needed to avoid semantic drift from `normalize_sequence_batch`

## 2. TIGER collate simplification

- [x] 2.1 Rename `collate_fn_train` to `collate_fn_sequence` and require configured input and attention mask fields
- [x] 2.2 Update `collate_fn_sequence` to stack preprocessed `input_ids`, `attention_mask`, optional `target_ids`, and optional output keys directly
- [x] 2.3 Remove `sequence_length` and `padding_token` usage from sequence collate; move inference normalization to preprocessing
- [x] 2.4 Confirm `normalize_sequence_batch` is no longer used by sequence collate

## 3. Configuration migration

- [x] 3.1 Add `normalize_sequence` after `generate_next_k_labels` in `configs/data/tiger_train.yaml` train preprocessing chain
- [x] 3.2 Add `normalize_sequence` after `generate_next_k_labels` in `configs/data/tiger_train.yaml` eval preprocessing chain
- [x] 3.3 Configure `normalize_sequence` with `input_field_name: input_ids`, `attention_mask_field_name: attention_mask`, `sequence_length: ${sequence_length}`, and `padding_token: -1`
- [x] 3.4 Update train/eval collate blocks to declare `attention_mask_field_name` and remove `sequence_length` from collate configuration
- [x] 3.5 Update inference preprocessing to declare `normalize_sequence` and inference collate to use `collate_fn_sequence`

## 4. Specification synchronization

- [x] 4.1 Ensure the new `preprocessed-sequence-normalization` spec matches the implemented `normalize_sequence` behavior
- [x] 4.2 Update TIGER preprocessing, collate, batch, sequence data, and config-declared preprocessing living specs through this change delta
- [x] 4.3 Check non-archive specs and configs for stale claims that TIGER sequence collate owns `sequence_length` or old collate entry points

## 5. Verification

- [x] 5.1 Run `uv run python -m compileall -q src/data/components/preprocessing.py src/data/components/collate.py src/data/utils.py`
- [x] 5.2 Run `uv run ruff check src/data/components/preprocessing.py src/data/components/collate.py src/data/utils.py`
- [x] 5.3 Run row-level `normalize_sequence` smoke for padding, trimming, attention mask generation, and target preservation
- [x] 5.4 Run `collate_fn_sequence` smoke confirming it only stacks preprocessed `input_ids` / `attention_mask` / `target_ids` and inference output keys
- [x] 5.5 Run `tiger_train` Hydra compose + datamodule instantiate smoke
- [x] 5.6 Run `openspec validate move-tiger-normalization-to-preprocessing --strict`
- [x] 5.7 Run `openspec validate --specs --no-interactive`
- [x] 5.8 Run `git diff --check`
