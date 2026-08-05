## Context

`Tiger` is now the self-contained LightningModule for TIGER training, evaluation, prediction, and generation. Its decoder-specific generation logic has been moved into `TigerDecoder`, but the Lightning step layer still contains a shallow `model_step` method that changes behavior based on whether `label_data` is present.

This hidden mode switch makes the interface hard to reason about: training and evaluation receive decoder hidden states plus loss, while prediction receives generated IDs plus a placeholder loss. The refactor should preserve the current runtime behavior while making each method's responsibility explicit.

## Goals / Non-Goals

**Goals:**
- Keep `forward()` as pure teacher-forcing model computation.
- Introduce an explicit helper for teacher-forcing loss computation.
- Make `training_step`, `eval_step`, and `predict_step` call the exact behavior they require.
- Preserve validation/test semantics: one teacher-forcing loss computation and one autoregressive generation pass.

**Non-Goals:**
- Do not change TIGER encoder/decoder architecture.
- Do not change loss definition, evaluator metrics, generated output format, or checkpoint compatibility beyond the existing active TIGER refactors.
- Do not add manual `train()` or `eval()` calls inside Lightning step methods.

## Decisions

1. Replace mode-dependent `model_step` with `_compute_loss`.
   - Rationale: loss computation has a stable interface requiring teacher-forcing decoder outputs and target IDs. Step methods remain responsible for invoking `forward()` so the computation flow stays explicit.
   - Alternative considered: keep `model_step` and rename flags. Rejected because it would preserve the hidden mode switch.

2. Let prediction call `generate()` directly.
   - Rationale: prediction has no loss contract and should only produce generated semantic IDs for output.
   - Alternative considered: create a generic `_generate` wrapper. Rejected unless it removes duplication; `generate()` already owns the model-level generation interface.

3. Keep Lightning mode management in hooks and trainer loops.
   - Rationale: Lightning sets train/eval mode around its loops. Step-local mode switching risks surprising nested behavior and makes hooks harder to reason about.
   - Alternative considered: call `self.train()` in `training_step` and `self.eval()` in `eval_step`. Rejected because it duplicates framework lifecycle behavior.

## Risks / Trade-offs

- [Risk] Removing `model_step` could break external direct callers if any exist. -> Mitigation: scan repository references and update all internal callers; this method is not part of the intended public runtime interface.
- [Risk] Evaluation remains computationally expensive because it computes loss and generation. -> Mitigation: document that this is intentional because validation/test need both teacher-forcing loss and ranking metrics.
