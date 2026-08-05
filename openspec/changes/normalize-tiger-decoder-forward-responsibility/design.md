## Context

`TigerDecoder` now owns decoder-side teacher-forcing and autoregressive generation. Its `forward()` method still accepts `future_ids=None` and uses that as a signal to run the BOS-only generation bootstrap. That hidden mode switch mirrors the earlier `Tiger.model_step` problem at the decoder level.

The desired interface is direct: `forward()` is teacher-forcing only, while `generate()` owns all autoregressive behavior, including the first BOS-only step.

## Goals / Non-Goals

**Goals:**
- Make `TigerDecoder.forward()` require future semantic IDs and only run teacher-forcing decoder computation.
- Move the BOS-only generation bootstrap into `TigerDecoder.generate()`.
- Keep generation input assembly inline in `generate()`.
- Preserve current generation and loss behavior.

**Non-Goals:**
- Do not introduce extra helper methods for input assembly or decoder execution.
- Do not change beam-search scoring, prefix validation, shared embedding ownership, or model architecture.
- Do not change `Tiger.forward()` or `Tiger.generate()` public behavior.

## Decisions

1. Make `future_ids` required in `TigerDecoder.forward()`.
   - Rationale: a required tensor makes the teacher-forcing contract explicit and prevents callers from triggering generation behavior through `forward()`.
   - Alternative considered: keep `future_ids | None` and document the branch. Rejected because the interface would remain mode-dependent.

2. Keep generation decoder input assembly inline in `generate()`.
   - Rationale: the user explicitly requested not to split out small helper functions. The generation path is local to one method and can remain readable with direct inline code.
   - Alternative considered: add `_run_decoder` and input assembly helpers. Rejected for this change to avoid over-fragmenting the module.

3. Let `generate()` call the wrapped HF decoder directly.
   - Rationale: generation should not call the teacher-forcing `forward()` interface after that interface is narrowed.
   - Alternative considered: use `self.forward()` for later generation steps only. Rejected because it would make generation depend on a teacher-forcing interface and keep mixed semantics.

## Risks / Trade-offs

- [Risk] Some future caller may expect BOS-only inference via `forward(future_ids=None)`. -> Mitigation: repository callers are updated to use `generate()` for autoregressive behavior.
- [Risk] Inline generation input assembly duplicates a small amount of `forward()` logic. -> Mitigation: this is accepted to keep the public method responsibilities explicit without introducing helper churn.
