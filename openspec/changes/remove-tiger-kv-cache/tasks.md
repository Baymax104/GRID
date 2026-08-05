## 1. Decoder Cache Removal

- [x] 1.1 Remove `DynamicCache`, `EncoderDecoderCache`, `past_key_values`, and `use_cache` handling from TIGER generation and decoder calls.
- [x] 1.2 Update beam search to return only generated IDs and marginal probabilities, without cache reorder.
- [x] 1.3 Simplify decoder input construction so generation recomputes from BOS/current prefix each hierarchy step.

## 2. Validation

- [x] 2.1 Scan TIGER code for residual KV cache references.
- [x] 2.2 Run focused lint for `src/recommendation/tiger`.
- [x] 2.3 Run OpenSpec validation for `remove-tiger-kv-cache`.
