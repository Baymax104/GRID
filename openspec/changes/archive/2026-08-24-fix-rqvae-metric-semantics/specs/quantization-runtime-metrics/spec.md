## ADDED Requirements

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
