## Context

TIGER generation currently passes `use_cache=True` into the HuggingFace decoder stack and threads `past_key_values` through beam search. The implementation is not fully coherent: the first decoder-returned cache can be replaced by a new empty `EncoderDecoderCache`, and the resulting code couples TIGER beam search to HuggingFace-specific cache classes.

The current refactor is already moving TIGER internals toward clearer package boundaries and more explicit encoder/decoder ownership. Removing KV cache keeps generation behavior straightforward while preserving generated semantic ID outputs.

## Goals / Non-Goals

**Goals:**

- Remove HuggingFace KV cache state from TIGER generation.
- Keep beam search ranking, prefix filtering, and generated SID tensor shapes unchanged.
- Keep training forward unchanged except for the simplified decoder signature.
- Reduce coupling to `DynamicCache` and `EncoderDecoderCache`.

**Non-Goals:**

- Introduce a replacement cache implementation.
- Optimize generation performance.
- Change beam search scoring, prefix validation, or top-k semantics.
- Change data preprocessing or model configuration.

## Decisions

- **Recompute decoder prefix each hierarchy step.** TIGER has a small `num_hierarchies`, so recomputing the current generated prefix is acceptable and much simpler than maintaining cache state across beam expansion.
- **Remove cache from public TIGER decoder calls.** `Tiger.decoder_forward_pass()` and `TigerDecoder.forward()` will return decoder embeddings only, without `(embeddings, past_key_values)` tuples.
- **Remove beam cache reorder.** Beam reordering will continue to reorder generated IDs, but no longer mutates decoder cache state.

## Risks / Trade-offs

- **Generation may be slightly slower** -> This is bounded by the small number of semantic ID hierarchies and avoids incorrect cache reuse.
- **Behavioral regressions in generation shape handling** -> Validate with lint and Hydra compose/import smoke; keep tensor path otherwise unchanged.
