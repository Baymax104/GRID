## Context

TIGER uses a shared SID embedding table whose indices are hierarchy-offset global SID tokens:

```text
global_sid = hierarchy * codebook_size + local_sid
```

The current decoder projection uses one local projection head per hierarchy. That works, but it is not symmetric with the input embedding space and makes the decoder output interface a set of local classification heads rather than one language-modeling head over the SID vocabulary.

## Goals / Non-Goals

**Goals:**
- Replace per-hierarchy projection heads with a single global SID `lm_head`.
- Return raw logits over the full global SID vocabulary.
- Convert local target IDs to global target IDs for loss.
- Slice global logits to the active hierarchy range during generation so generated IDs remain local per-hierarchy codes.

**Non-Goals:**
- Do not tie `lm_head.weight` to `sid_embedding_table.weight` in this change.
- Do not change generated output shape or evaluator inputs.
- Do not change beam-search prefix validation semantics.

## Decisions

1. Use one `lm_head` with output size `num_hierarchies * codebook_size`.
   - Rationale: this matches the global SID vocabulary used by the shared embedding table.
   - Alternative considered: one shared local head with output size `codebook_size`. Rejected because it collapses hierarchy-specific code semantics into one shared classifier.

2. Train against global SID targets.
   - Rationale: if logits are global, cross entropy targets must also be global. The local target IDs are offset by hierarchy before loss computation.
   - Alternative considered: slice logits before loss and keep local targets. Rejected because loss would no longer exercise the global vocabulary interface.

3. Slice logits during generation.
   - Rationale: generation still produces one local code for the current hierarchy at each step; beam search and prefix validation already operate on local IDs.
   - Alternative considered: let beam search operate directly on global IDs. Rejected because it would force wider downstream changes to generated ID shape and prefix validation.

## Risks / Trade-offs

- [Risk] The global head changes parameter names and shape, so existing per-hierarchy-head checkpoints are incompatible. -> Mitigation: checkpoint compatibility is not required for this refactor.
- [Risk] Global logits increase per-position output width from `codebook_size` to `num_hierarchies * codebook_size`. -> Mitigation: `num_hierarchies` is small, and generation slices only the active hierarchy range before beam search.
