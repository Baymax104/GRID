## Context

`collate_with_sid_causal_duplicate` currently performs three actions: normalize list rows into a dict batch, generate semantic-ID-aligned contiguous subsequences, optionally downsample them to `max_batch_size`, then call `collate_fn_train`. The wrapper is only referenced by `configs/data/tiger_train.yaml` train collate; eval/test already use `collate_fn_train`.

The desired shape is one public TIGER training collate entry point, with data augmentation controlled by configuration instead of by choosing a different function target.

## Goals / Non-Goals

**Goals:**
- Make `collate_fn_train` the only TIGER train/eval/test collate target.
- Move SID causal duplicate sampling into a helper with a narrow batch-in/batch-out contract.
- Preserve the existing sampling and masking behavior.
- Keep eval/test behavior unchanged by defaulting augmentation off.

**Non-Goals:**
- Do not change `TigerModelInput` / `TigerLabelData` semantics.
- Do not change `next_k_token_masking` or target label semantics.
- Do not change inference collate behavior.

## Decisions

1. **Use a helper for augmentation only**

   Add `sample_sid_causal_duplicate_sequences(batch, sequence_field_name, sid_hierarchy, max_batch_size)` in `src/data/components/collate.py`. The helper accepts an already-normalized `dict[str, list[Tensor]]` batch and returns the augmented batch. It does not perform padding, label generation, masking, or dataclass construction.

2. **Gate augmentation through `collate_fn_train`**

   Extend `collate_fn_train` with:

   ```python
   enable_sid_causal_duplicate: bool = False
   sequence_field_name: str | None = None
   sid_hierarchy: int | None = None
   max_batch_size: int = 128
   ```

   When `enable_sid_causal_duplicate` is true, `sequence_field_name` and `sid_hierarchy` are required. When false, these parameters are ignored.

3. **Remove wrapper entry point**

   Delete `collate_with_sid_causal_duplicate` after migrating `configs/data/tiger_train.yaml`. This prevents new call sites from depending on duplicate collate entry points.

4. **Keep train/eval config symmetric**

   `train_collate` and `eval_collate` both target `collate_fn_train`. Training sets `enable_sid_causal_duplicate: true`; eval/test omit the flag or set it false.

## Risks / Trade-offs

- **Risk: augmentation semantics drift** → Preserve the existing contiguous subsequence enumeration and sampling logic in the helper, and add smoke checks for augmented batch size and label shape.
- **Risk: missing required augmentation parameters** → Raise a clear `ValueError` when `enable_sid_causal_duplicate` is true but `sequence_field_name` or `sid_hierarchy` is missing.
- **Risk: eval/test accidentally augment** → Default `enable_sid_causal_duplicate` to false and keep eval/test config explicit or defaulted to false.
- **Risk: stale config target remains** → grep non-archive code/config/specs for `collate_with_sid_causal_duplicate`.
