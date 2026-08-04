## Why

Official train model configs currently expose training-only dependencies as separate sibling nodes (`loss_function`, `optimizer`, `scheduler`, and RQVAE's `reconstruction_loss_function`) and then mirror them into `root`. This keeps the primary model entrypoint noisy and spreads one conceptual group across multiple top-level fields.

## What Changes

- Introduce a small model training dependency config, `TrainingModelConfig`, for runtime training dependencies.
- Update train model configs to pass one `training_model_config` object into `model.root` instead of separate `loss_function` / `optimizer` / `scheduler` fields.
- Include RQVAE's `reconstruction_loss_function` in the same `training_model_config` object.
- Keep inference configs out of scope except for preserving their current behavior.
- Keep `model.root` as the Python-instantiated model entrypoint; do not reintroduce a `components` wrapper.
- **BREAKING**: Model constructor signatures for train-capable models change from separate training dependency parameters to a grouped `training_model_config` parameter.

## Capabilities

### New Capabilities

- `model-training-components`: Defines how official train model configs group loss, optimizer, scheduler, and optional reconstruction loss dependencies.

### Modified Capabilities

None.

## Impact

- Affected code:
  - `src/common/configs/model.py`
  - `src/quantization/rkmeans/residual_kmeans.py`
  - `src/quantization/rvq/residual_vector_quantization.py`
  - `src/quantization/rqvae/residual_quantization_vae.py`
  - `src/recommendation/tiger_generation_model.py`
- Affected config:
  - `configs/model/rkmeans_train.yaml`
  - `configs/model/rvq_train.yaml`
  - `configs/model/rqvae_train.yaml`
  - `configs/model/tiger_train.yaml`
- Affected tests: focused quantization model tests and Hydra config compose smoke checks.
- No dependency changes.
