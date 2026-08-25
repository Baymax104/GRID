## Context

GRID now supports `wandb://<run-id>` references for cross-stage artifacts such as semantic embeddings, semantic IDs, recommendation outputs, and checkpoints. The current resolver accepts short run URIs and tries to fill missing W&B identity from several implicit sources: `WANDB_ENTITY`, `WANDB_PROJECT`, the active `wandb.run`, and `wandb.Api().default_entity`.

That behavior makes identical configs behave differently across shells, machines, and distributed worker processes. It also conflicts with the current direction that W&B run lifecycle is owned by the configured Lightning `WandbLogger`, while artifact input identity should be declared in Hydra config rather than discovered from process state.

## Goals / Non-Goals

**Goals:**

- Make experiment-level W&B identity explicit as `user`, `project`, and `group`.
- Use `user` as W&B entity for both `WandbLogger` and W&B artifact reference resolution.
- Remove resolver fallbacks to environment variables, active W&B runs, and `wandb.Api().default_entity`.
- Keep fully-qualified W&B URIs (`wandb://<entity>/<project>/<run-id>`) working for cross-project references.
- Preserve local path behavior and rank-zero artifact download behavior.

**Non-Goals:**

- Do not change W&B Artifact selection semantics by role, alias, or file.
- Do not make artifact resolution depend on logger instantiation order.
- Do not add writer-level W&B run lifecycle fields.
- Do not introduce a new config namespace for W&B identity unless existing top-level identity fields prove insufficient.

## Decisions

1. Use top-level `user` as the W&B entity field.

   The user has already introduced `user: baymaxam` in an experiment config. Reusing that top-level field keeps the experiment entry as the visible source of W&B identity alongside existing `project` and `group`. Alternatives considered were `entity` and `wandb_entity`; `entity` is closer to W&B terminology but would diverge from the user's chosen field, while `wandb_entity` would reintroduce a W&B-prefixed top-level field similar to the previously removed `wandb_project`.

2. Keep Python artifact loader parameter names as `wandb_entity` and `wandb_project`.

   Component configs can map top-level `user` to callee parameter `wandb_entity`. This preserves the existing loader API shape and makes call sites self-documenting. The top-level experiment field remains concise, while implementation parameters remain explicit about W&B semantics.

3. Remove implicit identity fallbacks from `resolve_reference`.

   Short URI resolution should use only `uri.entity/project` or explicit `default_entity/default_project` arguments. This makes failure deterministic and prevents one rank or machine from succeeding only because it has W&B account defaults. The resolver may still use `wandb.Api()` to query a fully identified run path; only default identity discovery is removed.

4. Do not fix this by reordering logger/datamodule/model instantiation.

   `WandbLogger` initializes runs lazily when `logger.experiment` is accessed, and DDP rank behavior makes active-run-derived identity a poor input resolver contract. Config-provided identity is clearer and avoids coupling data artifact resolution to logger lifecycle.

5. Keep fully-qualified W&B URIs as an escape hatch.

   `wandb://baymaxam/GRID/<run-id>` should continue to bypass experiment defaults. This supports cross-project or cross-user artifact references without changing experiment-level W&B identity.

## Risks / Trade-offs

- [Risk] Existing user scripts that rely on `WANDB_ENTITY`, `WANDB_PROJECT`, or W&B default account state will fail for short URIs. -> Mitigation: provide a direct error telling users to set experiment `user/project` or use a fully-qualified URI.
- [Risk] Adding `user` to every official experiment creates a small amount of duplicated config. -> Mitigation: keep it visible at the experiment entry level because it is a manual run identity input like `project` and `group`.
- [Risk] W&B calls still require credentials even though identity is explicit. -> Mitigation: local paths continue to bypass W&B entirely; only `wandb://` references require W&B API access.
- [Risk] Tests may miss a config call site if only unit tests are updated. -> Mitigation: add Hydra compose checks for official configs that use W&B artifact loaders and loggers.
