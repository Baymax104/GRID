## 1. Streaming preprocessing pipeline

- [x] 1.1 Add row/result type aliases or helper functions for `None | row | Iterable[row]` preprocessing results in `src/data/datasets.py`
- [x] 1.2 Implement streaming flat-map preprocessing execution for `SequenceDataset`
- [x] 1.3 Ensure dictionary returns are treated as single rows rather than iterable keys
- [x] 1.4 Preserve existing `None` filtering behavior and infinite training iteration behavior

## 2. TIGER preprocessing helpers

- [x] 2.1 Add row-level SID causal duplicate expansion helper in `src/data/components/preprocessing.py`
- [x] 2.2 Implement lazy candidate selection for SID duplicate expansion without materializing all subsequences
- [x] 2.3 Add row-level TIGER next-k label generation helper that writes `input_ids` and `target_ids`
- [x] 2.4 Preserve existing masking semantics: first target position becomes `masking_token`, remaining target positions become `padding_token`
- [x] 2.5 Decide whether to delete or keep `src/data/components/label_functions.py` after label generation moves to preprocessing, then update references consistently

## 3. Collate simplification and configuration migration

- [x] 3.1 Simplify `collate_fn_train` to consume configured `input_field_name` and `target_field_name`
- [x] 3.2 Remove `label_generate_functions`, `masking_token`, and SID duplicate augmentation parameters from `collate_fn_train`
- [x] 3.3 Update `configs/data/tiger_train.yaml` train preprocessing chain to include SID expansion followed by label generation
- [x] 3.4 Update `configs/data/tiger_train.yaml` eval/test preprocessing chain to include label generation without SID expansion
- [x] 3.5 Update train/eval collate blocks to declare only input/target field names, `sequence_length`, and `padding_token`

## 4. Specification synchronization

- [x] 4.1 Update living specs for streaming preprocessing row expansion
- [x] 4.2 Update living specs for TIGER preprocessed label generation
- [x] 4.3 Update active change artifacts that mention collate-owned label generation or SID duplicate augmentation

## 5. Verification

- [x] 5.1 Run `uv run python -m compileall -q src/data/datasets.py src/data/components/preprocessing.py src/data/components/collate.py`
- [x] 5.2 Run `uv run ruff check src/data/datasets.py src/data/components/preprocessing.py src/data/components/collate.py`
- [x] 5.3 Run streaming dataset smoke for `None`, single-row, and generator row expansion preprocessing results
- [x] 5.4 Run SID duplicate preprocessing helper smoke for aligned subsequences and sampled max count
- [x] 5.5 Run TIGER label preprocessing smoke for `input_ids` and `target_ids` shapes and masking semantics
- [x] 5.6 Run `collate_fn_train` smoke confirming it only assembles preprocessed `input_ids` / `target_ids`
- [x] 5.7 Run `tiger_train` Hydra compose + datamodule instantiate smoke
- [x] 5.8 Run `openspec validate streaming-preprocessing-row-expansion --strict`
- [x] 5.9 Run `openspec validate --specs --no-interactive`
- [x] 5.10 Run `git diff --check`
