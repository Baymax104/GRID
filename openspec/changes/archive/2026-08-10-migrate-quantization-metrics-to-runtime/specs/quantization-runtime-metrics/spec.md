## ADDED Requirements

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
