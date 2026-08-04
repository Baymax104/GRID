## 1. Training Components Container

- [x] 1.1 Add `src/common/configs/model.py` with a passive `TrainingModelConfig` container for training-only dependencies.
- [x] 1.2 Keep the container free of optimizer construction, scheduler binding, loss execution, and model-specific training logic.

## 2. Model Constructor Migration

- [x] 2.1 Update `ResidualKMeans` to receive `training_model_config` and unpack existing runtime attributes from it.
- [x] 2.2 Update `ResidualVectorQuantization` to receive `training_model_config` and unpack existing runtime attributes from it.
- [x] 2.3 Update `ResidualQuantizationVAE` to receive `training_model_config`, including `reconstruction_loss_function`.
- [x] 2.4 Update `SemanticIDEncoderDecoder` to receive `training_model_config` and unpack existing runtime attributes from it.
- [x] 2.5 Update focused direct-instantiation tests to use the new constructor shape.

## 3. Config Migration

- [x] 3.1 Update `configs/model/rkmeans_train.yaml` to define `training_model_config` and pass it into `root`.
- [x] 3.2 Update `configs/model/rvq_train.yaml` to define `training_model_config` and pass it into `root`.
- [x] 3.3 Update `configs/model/rqvae_train.yaml` to define `training_model_config`, including reconstruction loss, and pass it into `root`.
- [x] 3.4 Update `configs/model/tiger_train.yaml` to define `training_model_config` and pass it into `root`.
- [x] 3.5 Confirm inference configs do not gain `training_model_config`.

## 4. Verification

- [x] 4.1 Run a residual scan confirming train `root` configs no longer pass separate `loss_function`, `optimizer`, `scheduler`, or `reconstruction_loss_function` fields.
- [x] 4.2 Run `openspec validate consolidate-model-training-components --strict`.
- [x] 4.3 Run focused `uv run ruff check` for touched common, model, and test files.
- [x] 4.4 Run Hydra compose smoke checks for affected train model configs.
- [x] 4.5 Run focused tests covering updated direct-instantiation paths.
