## Context

Official train model configs currently mount model configs under `@model` and expose the primary model object at `model.root`. The launcher instantiates `cfg.model.root` directly, so any grouping must preserve `root` as the primary model entrypoint and must not reintroduce the removed `components` wrapper.

The train-capable models all consume the same kind of training dependencies:

- primary loss function
- optimizer factory
- optional scheduler factory
- RQVAE-only reconstruction loss function

These dependencies are currently stored as separate top-level siblings in each train model config and mirrored into `root`, even though they form one conceptual training dependency group.

## Goals / Non-Goals

**Goals:**

- Group train-only model dependencies under one `training_model_config` object.
- Keep `model.root` as the only primary model object.
- Keep model code typed against a small runtime container instead of raw OmegaConf.
- Include RQVAE's `reconstruction_loss_function` in the same group.
- Preserve existing training behavior and optimizer/scheduler construction semantics.

**Non-Goals:**

- Do not modify inference model configs except to preserve existing behavior.
- Do not move trainer/callback/logger responsibilities into model config.
- Do not introduce shared base model classes or a shared quantization model abstraction.
- Do not change optimizer, scheduler, or loss hyperparameter values.

## Decisions

1. Introduce `TrainingModelConfig` under `src/common/configs/model.py`
   - Rationale: model training dependencies are a Hydra-instantiated runtime config model, not an executable component.
   - Alternative considered: Put it under `src/common/modules/`. Rejected because this is configuration/runtime dependency grouping, not an `nn.Module`.

2. Pass one `training_model_config` parameter to train-capable model constructors
   - Rationale: This removes field-by-field forwarding from config while keeping model constructors explicit about receiving train dependencies.
   - Alternative considered: Keep existing constructor fields and only group YAML nodes. Rejected because it keeps the Python API scattered.

3. Keep model internals using existing attributes
   - Rationale: Models can unpack `training_model_config` into `self.loss_function`, `self.optimizer`, and `self.scheduler`, minimizing changes to training logic and tests.
   - Alternative considered: Rewrite all training logic to dereference `self.training_model_config.*`. Rejected because it expands the implementation surface without behavior value.

4. Include `reconstruction_loss_function` in the same container
   - Rationale: It is a training loss dependency. Leaving it outside would keep RQVAE partially scattered.
   - Alternative considered: Keep reconstruction loss as a direct RQVAE constructor parameter. Rejected because the user explicitly wants it grouped too.

5. Represent absent schedulers as `scheduler: null` inside `training_model_config`
   - Rationale: RVQ and RQVAE currently pass `scheduler: null`; preserving that value inside the group avoids behavior changes.
   - Alternative considered: Omit the scheduler field for no-scheduler models. Rejected because a consistent field shape is easier to inspect and validate.

## Risks / Trade-offs

- [Risk] Constructor signature changes can break ad hoc Hydra overrides or direct test instantiation -> Mitigation: Update official configs and focused tests together; leave direct model behavior unchanged after unpacking.
- [Risk] A broad rename could touch inference configs unnecessarily -> Mitigation: Scope implementation to train model configs only.
- [Risk] The new grouping could be mistaken for a new model abstraction -> Mitigation: Keep it as a passive dataclass container with no training logic.
