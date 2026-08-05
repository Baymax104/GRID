## Why

TIGER loss computation currently reaches into `TigerDecoder.decoder_mlp` to project decoder hidden states into logits. This mixes model projection logic with loss aggregation and exposes decoder internals to the LightningModule.

## What Changes

- Move teacher-forcing logits projection into `TigerDecoder.forward()`.
- Make `TigerDecoder.forward()` return raw logits, not softmax probabilities.
- Keep `TigerDecoder.generate()` using the same decoder-side projection for autoregressive candidate logits.
- Update TIGER loss computation to consume logits and target IDs only.
- Preserve training, evaluation, and prediction behavior.

## Capabilities

### New Capabilities

### Modified Capabilities
- `self-contained-tiger-generation-model`: Clarify that decoder-side projection from hidden states to logits belongs to `TigerDecoder`, while loss aggregation belongs to `Tiger`.

## Impact

- Affected code: `src/recommendation/tiger/decoder.py`, `src/recommendation/tiger/tiger.py`.
- Affected behavior: internal model interface cleanup only; logits remain raw values suitable for cross entropy and beam-search scoring.
- Dependencies: none.
