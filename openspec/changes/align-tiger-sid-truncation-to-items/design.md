## Context

`tiger_train` maps item IDs to flattened semantic-ID token sequences before label generation. One item occupies `num_hierarchies` consecutive tokens. After label generation, `normalize_sequence` currently delegates to token-level trimming that keeps the most recent non-padding tokens up to `sequence_length`.

This can choose a left trim index that is not divisible by `num_hierarchies`, producing encoder input that starts in the middle of an item SID group. The resulting sequence no longer preserves hierarchy position semantics for the first retained item.

## Goals / Non-Goals

**Goals:**

- Preserve item SID boundaries when TIGER normalization trims flattened semantic-ID model inputs.
- Keep model input tensors fixed to `sequence_length`.
- Right-pad with `padding_token` when whole-item trimming leaves fewer than `sequence_length` real tokens.
- Keep `target_ids` unchanged and shaped as `(num_hierarchies,)` per row.
- Make SID hierarchy size explicit in `configs/data/tiger_train.yaml`.

**Non-Goals:**

- Do not change how semantic IDs are generated or looked up.
- Do not change `generate_next_k_labels` target selection semantics.
- Do not change `collate_fn_sequence` or TIGER model batch dataclasses.
- Do not apply item-boundary trimming to generic non-SID token sequences unless a hierarchy/group size is explicitly configured.

## Decisions

1. Add an explicit SID group-size parameter to row normalization.
   - Rationale: `normalize_sequence` currently has no way to know item boundaries. Passing `sid_hierarchy` or an equivalent `item_token_width` makes the behavior explicit at the TIGER config site.
   - Alternative considered: Infer hierarchy width from `target_ids.size(0)`. Rejected because normalization should not depend on labels being present forever, and config is clearer.

2. Trim by whole item groups before right-padding.
   - Rationale: If `sequence_length = 200` and `sid_hierarchy = 6`, keeping `floor(200 / 6) * 6 = 198` real tokens is better than keeping 200 tokens that include a partial leading item.
   - Alternative considered: Round up and keep an overlong sequence. Rejected because collate and model input contracts require fixed `sequence_length`.

3. Keep token-level normalization as the default behavior when no SID group size is configured.
   - Rationale: Existing normalization may be used by non-SID or future token sequences. The TIGER-specific behavior should be opt-in through config.
   - Alternative considered: Keep a separate `normalize_sequence_tensor` helper and add item-aware branching there. Rejected after moving data helpers into `src/data/utils.py`; the only remaining caller was `normalize_sequence`, so inlining keeps the behavior local to TIGER preprocessing.

4. Generate attention masks after final padding.
   - Rationale: The mask must reflect the actual fixed-length model input, with `1` for non-padding tokens and `0` for padding tokens.

## Risks / Trade-offs

- [Risk] Effective history length can decrease by up to `sid_hierarchy - 1` tokens when `sequence_length` is not divisible by `sid_hierarchy` -> Mitigation: this is intentional to preserve item integrity; document it as a breaking behavior change for long inputs.
- [Risk] Misconfigured group size could still corrupt boundaries -> Mitigation: validate configured group size is a positive integer and test `configs/data/tiger_train.yaml` passes `${num_hierarchies}` to normalization.
- [Risk] Existing tests may assert token-level trimming -> Mitigation: update/add focused tests for SID-aware trimming while preserving default token-level behavior without the group-size parameter.

## Migration Plan

1. Inline fixed-length normalization logic into `normalize_sequence`.
2. Add an optional item/SID token width parameter to `normalize_sequence`.
3. Set `sid_hierarchy: ${num_hierarchies}` on train and eval `normalize_sequence` entries in `configs/data/tiger_train.yaml`.
4. Add focused unit tests covering exact-fit, non-divisible length, short sequence padding, attention mask, and `target_ids` preservation.

## Open Questions

None.
