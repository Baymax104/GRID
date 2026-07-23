## ADDED Requirements

### Requirement: Quantization training configs SHALL hide internal loop hooks
Quantization training configuration files SHALL NOT expose internal training-loop hook fields that users are not expected to choose manually.

#### Scenario: User opens quantization training config
- **WHEN** 用户打开 RKMeans、RVQ 或 RQVAE training model config
- **THEN** config MUST NOT contain a `training_loop_function` field
- **AND** users MUST NOT need to understand manual optimization hooks to run quantization training
