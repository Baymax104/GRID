## Context

`configs/experiment/rkmeans_inference.yaml` already declares `project: GRID` and `group: rkmeans`, which is the intended shape. Most other experiments still declare `wandb_project: GRID`, while component configs independently set W&B logger and writer project/group values.

## Goals / Non-Goals

**Goals:**
- Make experiment configs the single source of truth for W&B project/group.
- Use experiment family names for group values.
- Keep component configs reusable by referencing `${project}` and `${group}`.
- Preserve existing W&B artifact helper function signatures.

**Non-Goals:**
- Do not change W&B URI syntax.
- Do not rename Python function parameters such as `wandb_project`.
- Do not add launcher-specific W&B branching.

## Decisions

1. Experiment-level `project` replaces `wandb_project`

   `project` is not W&B-specific in naming, but it maps directly to W&B `project` for loggers, writers, and artifact resolution defaults. This matches the existing `rkmeans_inference` shape.

2. `group` uses experiment family names

   Train/inference stages for one method share a group: `rkmeans`, `rqvae`, `rvq`, `tiger`, `sem_embeds`, or `tail_sid_diagnosis`. `task_name` remains the precise stage name and continues to be used for artifact names. W&B `job_type` records the lifecycle mode: `train`, `inference`, or `analysis`.

3. Component config references top-level identity

   Logger and writer configs use `project: ${project}` and `group: ${group}`. Data/model configs keep the callee parameter `wandb_project`, but set it to `${project}`.

4. Short W&B URI defaults read `cfg.project`

   `src.main` passes `cfg.get("project", None)` directly into checkpoint artifact resolution for train and inference runs. This keeps short W&B URI defaults tied to the experiment identity without adding a separate helper or fallback path.

## Risks / Trade-offs

- [Risk] External overrides still using `wandb_project=...` no longer affect repo configs -> use `project=...` instead.
- [Risk] Group names change from some `${task_name}` values to family names -> this is intended for clearer W&B experiment grouping.
