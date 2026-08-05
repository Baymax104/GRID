## Context

`Tiger` is a LightningModule. Lightning already switches the module tree between train and eval mode for training, validation, test, and prediction loops. The current TIGER implementation adds a `_make_deterministic()` helper that manually calls `.train()` and `.eval()` on wrapped Hugging Face encoder/decoder submodules and sets a local `is_training` attribute.

Repository search shows `is_training` is not consumed outside those hooks. The custom mode hooks therefore duplicate framework lifecycle behavior while adding a second place to reason about module mode.

## Goals / Non-Goals

**Goals:**
- Remove TIGER-specific manual train/eval switching hooks.
- Keep metric reset and metric logging hooks unchanged.
- Let Lightning own mode switching for standard trainer loops.

**Non-Goals:**
- Do not change generation, loss, evaluator, encoder, or decoder logic.
- Do not alter metric reset timing.
- Do not introduce direct `train()` or `eval()` calls in step methods.

## Decisions

1. Delete `_make_deterministic()` and hooks whose only purpose is mode switching.
   - Rationale: Lightning recursively manages module mode for its loops, and the repository does not use the custom `is_training` attribute.
   - Alternative considered: keep the helper but remove `is_training`. Rejected because manual `.train()` / `.eval()` calls would still duplicate Lightning.

2. Keep lifecycle hooks that reset or log metrics.
   - Rationale: those hooks are domain behavior, not mode switching.
   - Alternative considered: consolidate metric hooks in the same change. Rejected as unrelated cleanup.

## Risks / Trade-offs

- [Risk] Direct calls to `model.generate()` outside Lightning will no longer receive mode changes from these hooks. -> Mitigation: direct inference callers should use the standard PyTorch contract: call `model.eval()` and run under `torch.no_grad()` when needed; the removed hooks never affected direct calls anyway.
- [Risk] A hidden dependency on the custom `is_training` attribute could exist outside the repository. -> Mitigation: no repository references exist; external use would be relying on an undocumented local attribute.
