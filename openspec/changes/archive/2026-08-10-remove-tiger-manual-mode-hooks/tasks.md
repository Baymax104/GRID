## 1. Remove Manual Mode Switching

- [x] 1.1 Remove TIGER's `_make_deterministic` helper.
- [x] 1.2 Remove validation/test/predict hooks that only call `_make_deterministic`.
- [x] 1.3 Remove `_make_deterministic` calls from retained metric lifecycle hooks.

## 2. Preserve Metric Lifecycle

- [x] 2.1 Keep train, validation, and test metric reset behavior intact.
- [x] 2.2 Confirm step methods do not call `train()` or `eval()` directly.

## 3. Validation

- [x] 3.1 Scan for stale `_make_deterministic` and custom `is_training` references.
- [x] 3.2 Run focused lint for `src/recommendation/tiger`.
- [x] 3.3 Run OpenSpec validation for `remove-tiger-manual-mode-hooks`.
- [x] 3.4 Run TIGER Hydra compose/import smoke.
