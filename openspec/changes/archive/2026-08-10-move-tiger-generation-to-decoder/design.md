## Context

The current TIGER refactor has separated the model into a `src/recommendation/tiger/` package, but decoder-specific behavior is still split across `Tiger` and `TigerDecoder`. `Tiger` owns decoder input assembly, beam search, generated prefix expansion, and prefix validation, while `TigerDecoder` only wraps the underlying transformer decoder call.

The desired boundary is for `Tiger` to orchestrate the end-to-end Lightning workflow and encoder/decoder interaction, while `TigerDecoder` owns decoder-side teacher-forcing and autoregressive generation behavior.

## Goals / Non-Goals

**Goals:**

- Move decoder input assembly into `TigerDecoder.forward()`.
- Move autoregressive generation and beam-search step logic into `TigerDecoder.generate()`.
- Move decoder-side prefix validation state and logic into `TigerDecoder`.
- Preserve `Tiger` ownership of the shared SID embedding table by passing it to both encoder and decoder.
- Preserve output shapes and generation semantics.

**Non-Goals:**

- Change the Hydra `_target_` for TIGER.
- Change beam scoring, top-k semantics, prefix validity semantics, or evaluator behavior.
- Reintroduce KV cache.
- Change data preprocessing.

## Decisions

- **Keep `Tiger` as the Lightning orchestration module.** `Tiger` remains responsible for training hooks, loss logging, evaluator calls, and invoking encoder before decoder.
- **Make `TigerDecoder` a SID-aware decoder.** It will receive `sid_embedding_table`, `codebook_size`, `num_hierarchies`, `semantic_ids`, `top_k_for_generation`, and `should_check_prefix` so it can embed future IDs and run beam search internally.
- **Keep embedding table registered on `Tiger`.** The shared `nn.Embedding` remains a direct `Tiger` attribute and is passed by reference to `TigerEncoder` and `TigerDecoder`; this keeps ownership explicit and avoids each submodule creating separate tables.
- **Do not reintroduce cache.** Decoder generation recomputes from the current prefix each hierarchy step, matching the current post-cache-removal behavior.

## Risks / Trade-offs

- **More constructor parameters on `TigerDecoder`** -> This is acceptable because the decoder now owns a deeper domain behavior rather than acting as a generic wrapper.
- **Risk of changing generation shapes during the move** -> Validate with focused lint, residual reference scans, and Hydra compose/import smoke for `tiger_train` and `tiger_inference`.
