## ADDED Requirements

### Requirement: Inactive model implementation toggles SHALL stay out of config
Model configuration files SHALL NOT expose implementation toggles that are unused by all executable experiments and always set to the default no-op value.

#### Scenario: Quantization model config removes inactive CPU initialization toggle
- **WHEN** an implementation toggle such as `initialize_on_cpu` is not enabled by any executable experiment or script
- **THEN** the model config MUST omit that field
- **AND** users MUST NOT need to decide whether to set the inactive toggle
