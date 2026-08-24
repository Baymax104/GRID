# quantization-runtime-metrics Specification

## Purpose
TBD - created by archiving change migrate-quantization-metrics-to-runtime. Update Purpose after archive.
## Requirements
### Requirement: Quantization models SHALL expose metric payloads
RKMeans, RVQ, and RQVAE SHALL return metric payloads from train, validation, and test steps instead of managing framework metrics inside the model.

#### Scenario: Train step returns required loss payload
- **WHEN** a quantization training step runs
- **THEN** it MUST return a mapping containing `loss`
- **THEN** it MUST NOT call `self.log_dict` for framework-managed metrics

#### Scenario: Eval steps return scalar stat payloads
- **WHEN** validation or test step runs
- **THEN** it MUST return a mapping containing `loss`, `first_residuals_norm_ratio`, `last_residuals_norm_ratio`, `frac_unique_ids`, and `mse`
- **THEN** it MUST NOT reset or log framework-managed metrics directly

### Requirement: Quantization train configs SHALL declare runtime metrics
Official quantization train configs SHALL declare metrics under `model.metrics`.

#### Scenario: Train configs declare scalar metrics
- **WHEN** a maintainer inspects quantization train model configs
- **THEN** each config MUST declare train, validation, and test scalar metrics under `model.metrics`

#### Scenario: Train configs declare per-layer repeat metrics
- **WHEN** a maintainer inspects quantization train model configs
- **THEN** train metrics MUST use repeat definitions for per-layer coverage and entropy
- **THEN** repeat count MUST derive from `${num_hierarchies}`

### Requirement: Quantization training steps SHALL return complete metric payloads every step
Quantization training steps SHALL compute and return all configured train metric payload fields every step without gating metric statistics on `trainer.log_every_n_steps`.

#### Scenario: Train step does not gate metric payload fields
- **WHEN** a quantization training step runs
- **THEN** it MUST compute output statistics for that batch
- **THEN** it MUST return all configured train metric payload fields
- **THEN** it MUST NOT branch on `trainer.log_every_n_steps`

### Requirement: Quantization train and validation loss curves SHALL use comparable logging windows
Official quantization training metrics SHALL rely on the metric runtime's non-cumulative train logging semantics so train loss curves can be compared with validation loss windows without mixing full-training cumulative aggregates and validation-window aggregates.

#### Scenario: Maintainer compares train and validation loss curves
- **WHEN** a quantization training run logs `train/loss` and `val/loss`
- **THEN** `train/loss` MUST represent the latest logged train batch or logging window
- **AND** `val/loss` MUST represent the current validation window
- **AND** neither curve MUST be interpreted as a cumulative aggregate from training start

### Requirement: RQ-VAE eval payloads SHALL expose decomposed losses
ResidualQuantizationVAE SHALL return validation and test payloads whose `loss` field has the same weighted total loss semantics as training, and SHALL expose quantization and reconstruction loss components separately.

#### Scenario: RQ-VAE validation returns total and component losses
- **WHEN** `ResidualQuantizationVAE.validation_step` runs after the model can compute reconstruction loss
- **THEN** the returned payload MUST contain `loss`, `quantization_loss`, and `reconstruction_loss`
- **AND** `loss` MUST equal `quantization_loss_weight * quantization_loss + reconstruction_loss_weight * reconstruction_loss`
- **AND** `loss` MUST NOT represent quantization loss alone

#### Scenario: RQ-VAE test returns total and component losses
- **WHEN** `ResidualQuantizationVAE.test_step` runs after the model can compute reconstruction loss
- **THEN** the returned payload MUST contain `loss`, `quantization_loss`, and `reconstruction_loss`
- **AND** `loss` MUST equal `quantization_loss_weight * quantization_loss + reconstruction_loss_weight * reconstruction_loss`

#### Scenario: RQ-VAE config declares validation and test component loss metrics
- **WHEN** a maintainer inspects `configs/model/rqvae_train.yaml`
- **THEN** the validation metric config MUST declare scalar metrics for `loss`, `quantization_loss`, and `reconstruction_loss`
- **AND** the test metric config MUST declare scalar metrics for `loss`, `quantization_loss`, and `reconstruction_loss`

### Requirement: RQ-VAE residual and reconstruction metrics SHALL use explicit spaces
ResidualQuantizationVAE SHALL distinguish encoded-space quantization statistics from reconstruction-space statistics in metric payload field names and calculations.

#### Scenario: Encoded residual metrics use encoded-space normalization
- **WHEN** RQ-VAE computes residual statistics from encoded residual tensors
- **THEN** encoded residual norm ratios MUST use the encoded embedding norm as the denominator
- **AND** encoded residual metrics MUST use names that identify the encoded space, such as `encoded_first_residuals_norm_ratio`, `encoded_last_residuals_norm_ratio`, and `encoded_mse`
- **AND** encoded residual metrics MUST NOT divide by the original input embedding norm

#### Scenario: Reconstruction metrics compare decoder output with normalized input
- **WHEN** RQ-VAE computes reconstruction statistics
- **THEN** reconstruction metrics MUST compare decoder output against the normalized input embedding
- **AND** reconstruction metrics MUST use names that identify the reconstruction space, such as `reconstruction_mse` or `reconstruction_norm_ratio`

#### Scenario: RQ-VAE config declares space-explicit scalar metrics
- **WHEN** a maintainer inspects `configs/model/rqvae_train.yaml`
- **THEN** train, validation, and test metric configs MUST declare space-explicit RQ-VAE scalar metrics
- **AND** ambiguous RQ-VAE residual metric names MUST NOT be the only available metric names for judging quantization or reconstruction quality

