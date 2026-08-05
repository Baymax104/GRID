## Why

`TigerDecoder.forward()` currently has a mode-dependent branch: with `future_ids` it runs teacher-forcing, and without `future_ids` it bootstraps generation from BOS only. This makes the decoder forward interface ambiguous and mixes autoregressive generation behavior into the teacher-forcing path.

## What Changes

- Make `TigerDecoder.forward()` a teacher-forcing-only computation path.
- Require `future_ids` for `TigerDecoder.forward()` and remove the BOS-only branch from it.
- Keep BOS-only generation bootstrap inside `TigerDecoder.generate()`.
- Do not introduce extra small helper functions; keep the generation input assembly inline in `generate()`.
- Preserve generated IDs, marginal probabilities, and training loss behavior.

## Capabilities

### New Capabilities

### Modified Capabilities
- `self-contained-tiger-generation-model`: Clarify decoder forward and generation responsibilities.

## Impact

- Affected code: `src/recommendation/tiger/decoder.py`.
- Affected behavior: internal method responsibility only; external TIGER training/evaluation/prediction behavior remains equivalent.
- Dependencies: none.
