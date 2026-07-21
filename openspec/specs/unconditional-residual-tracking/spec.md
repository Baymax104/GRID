# unconditional-residual-tracking Specification

## Purpose
TBD - created by archiving change remove-track-residuals-config. Update Purpose after archive.
## Requirements
### Requirement: Residual quantizers always expose per-layer residuals
The system SHALL make `ResidualKMeans`, `ResidualVectorQuantization`, and `ResidualQuantizationVAE` collect the residual after every quantization layer and return the stacked tensor in their existing residual output position. The returned tensor MUST have shape `(batch_size, n_features, n_layers)`.

#### Scenario: Quantizer forward pass
- **WHEN** any supported residual quantizer processes a batch with `n_layers` quantization layers
- **THEN** its residual output is a tensor containing one post-layer residual for each layer in layer order

### Requirement: Residual tracking is not configurable
The system SHALL NOT expose a `track_residuals` constructor parameter, instance setting, or model YAML configuration key for supported residual quantizers.

#### Scenario: Compose a quantizer experiment configuration
- **WHEN** Hydra composes any residual quantizer experiment configuration
- **THEN** the composed model configuration contains no `track_residuals` key and quantizer construction still succeeds

#### Scenario: Construct a quantizer directly
- **WHEN** a caller constructs a supported residual quantizer using its public constructor
- **THEN** the constructor signature does not accept a `track_residuals` argument

