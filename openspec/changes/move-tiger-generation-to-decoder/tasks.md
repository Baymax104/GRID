## 1. Decoder Ownership

- [x] 1.1 Extend `TigerDecoder` constructor to receive shared SID embedding table, codebook size, hierarchy count, generation top-k, semantic ID tensor, and prefix-check flag.
- [x] 1.2 Move semantic ID embedding helper and decoder input assembly into `TigerDecoder`.
- [x] 1.3 Move prefix validation and beam-search step logic into `TigerDecoder`.
- [x] 1.4 Add `TigerDecoder.generate()` for autoregressive semantic ID generation.

## 2. Tiger Simplification

- [x] 2.1 Update `Tiger` construction to pass shared decoder dependencies into `TigerDecoder`.
- [x] 2.2 Simplify `Tiger.forward()` to call `TigerDecoder.forward()`.
- [x] 2.3 Simplify `Tiger.generate()` to call `TigerDecoder.generate()` after encoder execution.
- [x] 2.4 Remove decoder-specific helper methods from `Tiger`.

## 3. Validation

- [x] 3.1 Scan for stale `decoder_forward_pass`, `_beam_search_one_step`, and decoder-side prefix helper references in `Tiger`.
- [x] 3.2 Run focused lint for `src/recommendation/tiger`.
- [x] 3.3 Run OpenSpec validation for `move-tiger-generation-to-decoder`.
- [x] 3.4 Run TIGER Hydra compose/import smoke.
