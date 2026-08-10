## 1. TIGER Step Interface

- [x] 1.1 Replace mode-dependent `model_step` with an explicit `_compute_loss` helper for teacher-forcing decoder outputs and target IDs.
- [x] 1.2 Update `training_step` to call `_compute_loss` directly and preserve train loss logging.
- [x] 1.3 Update `eval_step` to compute loss and call `generate()` explicitly once for evaluator metrics.
- [x] 1.4 Update `predict_step` to call `generate()` directly and return `ModelOutput` without placeholder loss handling.

## 2. Cleanup

- [x] 2.1 Remove stale internal references to `model_step`.
- [x] 2.2 Confirm step methods do not call `train()` or `eval()` directly.

## 3. Validation

- [x] 3.1 Run focused lint for `src/recommendation/tiger`.
- [x] 3.2 Run OpenSpec validation for `normalize-tiger-step-responsibilities`.
- [x] 3.3 Run TIGER Hydra compose/import smoke.
