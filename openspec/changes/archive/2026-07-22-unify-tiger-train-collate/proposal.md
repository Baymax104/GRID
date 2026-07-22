## Why

TIGER training currently has two training collate entry points: `collate_fn_train` for normal masking and `collate_with_sid_causal_duplicate` for training-time SID causal duplicate sampling. The second function is only a wrapper around augmentation plus `collate_fn_train`, which makes configuration and maintenance more complex than necessary.

## What Changes

- Extract the SID causal duplicate sampling logic into a dedicated helper function, tentatively `sample_sid_causal_duplicate_sequences(...)`.
- Add an explicit `enable_sid_causal_duplicate: bool = False` switch to `collate_fn_train`.
- Move SID causal duplicate parameters (`sequence_field_name`, `sid_hierarchy`, `max_batch_size`) onto `collate_fn_train` and use them only when augmentation is enabled.
- Update `configs/data/tiger_train.yaml` so both train and eval/test collate blocks use `src.data.components.collate.collate_fn_train`.
- Delete `collate_with_sid_causal_duplicate` as a public collate entry point.
- Preserve the existing augmentation semantics: enumerate contiguous semantic-ID-aligned subsequences containing at least two items, sample at most `max_batch_size`, and copy non-sequence fields from the original row.
- **BREAKING**: `src.data.components.collate.collate_with_sid_causal_duplicate` is removed; configuration must use `collate_fn_train` with `enable_sid_causal_duplicate=true` for training augmentation.

## Capabilities

### New Capabilities
- `unified-tiger-train-collate`: Defines `collate_fn_train` as the single TIGER train/eval/test collate entry point with optional SID causal duplicate augmentation.

### Modified Capabilities
- `tiger-specific-batch-contract`: TIGER training collate behavior now includes optional SID causal duplicate augmentation behind an explicit flag.
- `tiger-sequence-data-contract`: TIGER sequence data configuration must use `collate_fn_train` for train/eval/test, with augmentation enabled only for training.

## Impact

- Code: `src/data/components/collate.py`.
- Config: `configs/data/tiger_train.yaml`.
- Specs: new `unified-tiger-train-collate`, modified `tiger-specific-batch-contract` and `tiger-sequence-data-contract`.
- Verification: grep for removed wrapper target, compile/ruff, train collate smoke with augmentation enabled, eval collate smoke with augmentation disabled, Hydra compose/instantiate for `tiger_train`, OpenSpec validation, and `git diff --check`.
