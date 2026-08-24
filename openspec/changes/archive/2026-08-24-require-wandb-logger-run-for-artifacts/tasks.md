## 1. Writer Run Ownership

- [x] 1.1 Add a focused helper for retrieving the configured W&B logger-owned run from Lightning trainer loggers.
- [x] 1.2 Refactor `WandbArtifactWriter` to publish only through the logger-owned run and remove `wandb.init`, `finish_run`, run metadata parameters, and `fail_on_error`.
- [x] 1.3 Refactor `WandbCheckpointWriter` to publish only through the logger-owned run and remove `wandb.init`, `finish_run`, run metadata parameters, and `fail_on_error`.
- [x] 1.4 Update `WandbArtifactLineageCallback` to record lineage through the logger-owned run while preserving `fail_on_missing_run` behavior.

## 2. Configuration Cleanup

- [x] 2.1 Remove writer-level run lifecycle fields from all official `wandb_artifact_writer` callback configs.
- [x] 2.2 Remove writer-level run lifecycle fields from all official `wandb_checkpoint_writer` callback configs.
- [x] 2.3 Confirm official train, inference, and analysis W&B-backed experiments keep W&B run identity in `configs/logger/*.yaml`.
- [x] 2.4 Remove stale OpenSpec or documentation claims that W&B writers can independently create artifact-only runs.

## 3. Tests and Validation

- [x] 3.1 Replace writer tests for self-created W&B runs with tests proving logger-owned run publishing and absence of `wandb.init`/`finish`.
- [x] 3.2 Add tests proving missing logger-owned runs fail for W&B artifact and checkpoint writers.
- [x] 3.3 Update lineage callback tests for logger-owned run lookup and missing-run behavior.
- [x] 3.4 Add config assertions that W&B writer callback configs do not contain run lifecycle or `fail_on_error` fields.
- [x] 3.5 Run focused writer/lineage tests, relevant Hydra compose checks, `uv run ruff check src tests`, `uv run pytest`, and `openspec validate require-wandb-logger-run-for-artifacts --strict`.
