## 1. Decoder Projection

- [x] 1.1 Replace per-hierarchy `decoder_mlp` modules with one global `lm_head`.
- [x] 1.2 Update `TigerDecoder.forward()` to return raw global SID logits.
- [x] 1.3 Update `TigerDecoder.generate()` to slice active hierarchy logits before beam search.

## 2. Loss Targets

- [x] 2.1 Convert local `target_ids` to hierarchy-offset global targets in TIGER loss computation.
- [x] 2.2 Update TIGER logits/loss docstrings and shape comments.

## 3. Validation

- [x] 3.1 Scan for stale `decoder_mlp` references and unintended weight tying.
- [x] 3.2 Run focused lint for `src/recommendation/tiger`.
- [x] 3.3 Run OpenSpec validation for `use-global-tiger-lm-head`.
- [x] 3.4 Run TIGER Hydra compose/import smoke.
