## 1. Decoder Forward Contract

- [x] 1.1 Make `TigerDecoder.forward()` require `future_ids` and remove its BOS-only `future_ids is None` branch.
- [x] 1.2 Keep teacher-forcing input assembly in `TigerDecoder.forward()` using BOS plus future semantic ID embeddings.
- [x] 1.3 Update type hints and docstring to reflect teacher-forcing-only behavior.

## 2. Decoder Generation Path

- [x] 2.1 Update `TigerDecoder.generate()` so the first hierarchy step constructs BOS-only decoder inputs inline.
- [x] 2.2 Update later generation steps to construct BOS plus generated semantic ID embeddings inline.
- [x] 2.3 Ensure `TigerDecoder.generate()` no longer calls `TigerDecoder.forward()`.

## 3. Validation

- [x] 3.1 Scan for stale `forward(future_ids=None)` behavior and `self.forward()` calls inside `TigerDecoder.generate()`.
- [x] 3.2 Run focused lint for `src/recommendation/tiger`.
- [x] 3.3 Run OpenSpec validation for `normalize-tiger-decoder-forward-responsibility`.
- [x] 3.4 Run TIGER Hydra compose/import smoke.
