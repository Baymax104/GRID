## 1. Config Identity

- [x] 1.1 Add top-level `user: baymaxam` to all official W&B-backed experiment configs.
- [x] 1.2 Add `entity: ${user}` to all official W&B logger component configs.
- [x] 1.3 Add `wandb_entity: ${user}` beside existing `wandb_project: ${project}` in all data/model artifact loader configs.
- [x] 1.4 Confirm `rkmeans_train`, `rvq_train`, `rqvae_train`, `tiger_train`, inference configs, and analysis configs compose with `user/project/group`.

## 2. Explicit Resolver Semantics

- [x] 2.1 Remove `active_wandb_attr` and `default_wandb_entity` fallback helpers from `src/utils/wandb.py`.
- [x] 2.2 Update `resolve_reference` to use only URI-provided `entity/project` or explicit `default_entity/default_project` arguments.
- [x] 2.3 Update short URI error text to mention experiment `user/project` defaults or fully-qualified W&B URIs.
- [x] 2.4 Preserve local path bypass behavior and fully-qualified `wandb://<entity>/<project>/<run-id>` behavior.

## 3. Entrypoint Integration

- [x] 3.1 Update training checkpoint resolution to pass `default_entity=cfg.get("user", None)` and `default_project=cfg.get("project", None)`.
- [x] 3.2 Update inference checkpoint resolution to pass `default_entity=cfg.get("user", None)` and `default_project=cfg.get("project", None)`.
- [x] 3.3 Avoid changing launcher order solely to satisfy W&B artifact identity resolution.
- [x] 3.4 Preserve user-owned edits already present in `configs/experiment/rkmeans_train.yaml` and `src/utils/launcher.py`.

## 4. Tests and Verification

- [x] 4.1 Update artifact resolver tests to assert no environment variable, active run, or `wandb.Api().default_entity` fallback is used.
- [x] 4.2 Update run dispatch tests for checkpoint resolver calls to include explicit `default_entity`.
- [x] 4.3 Add or update Hydra compose tests confirming official W&B-backed configs wire `${user}` into logger entity and artifact loader `wandb_entity`.
- [x] 4.4 Run `uv run ruff check src tests`.
- [x] 4.5 Run focused tests for artifact resolution, main run modes, and config composition.
- [x] 4.6 Run `openspec validate require-explicit-wandb-entity --strict`.
