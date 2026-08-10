## Why

TIGER currently embeds semantic IDs in a global hierarchy-offset vocabulary but projects decoder hidden states with one local head per hierarchy. This is asymmetric: the input side treats hierarchy-specific codes as distinct global tokens, while the output side treats them as separate local classification tasks.

## What Changes

- Replace per-hierarchy decoder projection heads with one global SID `lm_head`.
- Make decoder teacher-forcing forward return raw logits over the global SID vocabulary.
- Convert local target IDs to global SID target IDs when computing loss.
- During generation, slice the global logits to the current hierarchy's codebook range before beam-search scoring.
- Do not tie `lm_head` weights to the shared SID embedding table in this change.

## Capabilities

### New Capabilities

### Modified Capabilities
- `self-contained-tiger-generation-model`: Align TIGER decoder logits with the global SID vocabulary used by SID embeddings.

## Impact

- Affected code: `src/recommendation/tiger/decoder.py`, `src/recommendation/tiger/tiger.py`.
- Affected behavior: internal projection parameterization changes; generated IDs remain local per-hierarchy SID codes and external outputs remain unchanged.
- Checkpoint compatibility: existing checkpoints with per-hierarchy `decoder_mlp` weights will not be compatible with the new `lm_head` parameter names and shape.
- Dependencies: none.
