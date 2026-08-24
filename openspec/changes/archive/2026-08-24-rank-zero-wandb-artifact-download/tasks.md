## 1. Download Helper

- [x] 1.1 Add a helper in `src/utils/wandb.py` that downloads artifacts once per distributed run.
- [x] 1.2 Update `resolve_wandb_artifact(...)` to use the helper and preserve existing metadata behavior.

## 2. Tests

- [x] 2.1 Cover non-distributed behavior still calling `artifact.download(...)`.
- [x] 2.2 Cover distributed rank 0 calling `artifact.download(...)` and barrier.
- [x] 2.3 Cover distributed non-zero rank skipping `artifact.download(...)` and resolving from download root.

## 3. Validation

- [x] 3.1 Run focused tests, ruff, and strict OpenSpec validation.
