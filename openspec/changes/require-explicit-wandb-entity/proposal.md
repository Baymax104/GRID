## Why

Short W&B run references such as `wandb://01mw1fez` currently depend on implicit entity discovery from environment variables, an active W&B run, or `wandb.Api().default_entity`. That makes artifact resolution sensitive to process order and machine-specific W&B state, which surfaced as `rvq_train` failing while an apparently similar `rkmeans_train` setup could pass under a different environment.

## What Changes

- **BREAKING**: Short W&B URI resolution MUST require explicit experiment-provided W&B identity; resolver code MUST NOT infer entity or project from `WANDB_ENTITY`, `WANDB_PROJECT`, an active run, or `wandb.Api().default_entity`.
- Official experiment configs SHALL declare top-level `user`, `project`, and `group` for W&B identity.
- Logger configs SHALL pass `entity: ${user}` and `project: ${project}` to `WandbLogger`.
- Artifact loader configs for `embedding_path`, `semantic_id_path`, and similar run references SHALL pass `wandb_entity: ${user}` and `wandb_project: ${project}`.
- Checkpoint W&B URI resolution in main run dispatch SHALL pass `default_entity=${user}` and `default_project=${project}`.
- Error messages and tests SHALL reflect that short W&B URIs require explicit config identity or a fully-qualified URI.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `experiment-config-componentization`: W&B identity ownership expands from `project/group` to `user/project/group`, and component configs must reference `user` when they need W&B entity.
- `wandb-artifact-lineage`: W&B reference resolver defaults must come only from explicit config arguments or fully-qualified URIs, not environment variables or W&B API default entity discovery.
- `keyed-prediction-bundle-artifact`: Bundle artifact input references using short W&B URIs must resolve with explicit experiment `user/project` defaults.
- `model-training-components`: Checkpoint W&B URI resolution must use explicit experiment `user/project` defaults.

## Impact

- Affected configs: `configs/experiment/*.yaml`, `configs/logger/*.yaml`, data/model configs that call artifact loaders.
- Affected code: `src/data/components/artifacts.py`, `src/utils/wandb.py`, `src/main.py`.
- Affected tests: artifact resolver tests, run dispatch tests, and lightweight Hydra config composition checks.
- External behavior: users must configure `user` or use fully-qualified `wandb://<entity>/<project>/<run-id>` URIs; relying on shell W&B environment variables or W&B account defaults is no longer supported.
