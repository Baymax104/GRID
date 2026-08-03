## Why

Current TIGER sequence normalization trims long flattened semantic-ID inputs by token count. Because each item is represented by `num_hierarchies` consecutive SID tokens, token-level left trimming can start in the middle of an item SID and feed the model hierarchy-misaligned encoder input.

## What Changes

- Change TIGER input normalization so long flattened SID sequences are left-trimmed by whole item groups, not arbitrary SID tokens.
- Preserve fixed-length model inputs by right-padding with `padding_token` after item-aligned trimming when the retained whole-item token count is shorter than `sequence_length`.
- Make the item grouping parameter explicit in TIGER preprocessing configuration, using `num_hierarchies`/`sid_hierarchy` for SID-aware normalization.
- Keep `target_ids` unchanged; normalization continues to affect only the configured model input field and generated attention mask.
- **BREAKING**: Long TIGER encoder inputs may contain fewer real tokens than before when `sequence_length` is not divisible by `num_hierarchies`, because partial item SID groups must be discarded rather than kept.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `preprocessed-sequence-normalization`: Update TIGER normalization semantics so flattened SID inputs are truncated on item boundaries and padded back to fixed `sequence_length`.

## Impact

- Affected code: `src/data/components/preprocessing.py`.
- Affected config: `configs/data/tiger_train.yaml`.
- Affected tests: focused unit coverage for item-aligned truncation, padding, attention masks, and target-label preservation.
- No dependency changes.
