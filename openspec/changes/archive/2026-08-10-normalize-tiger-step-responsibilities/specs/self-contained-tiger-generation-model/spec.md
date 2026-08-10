## ADDED Requirements

### Requirement: TIGER Lightning steps SHALL use explicit train, evaluation, and prediction paths
TIGER Lightning step methods SHALL call explicit computation paths for their runtime role instead of relying on a mode-dependent `model_step` dispatcher. `forward()` SHALL remain a pure teacher-forcing computation path and MUST NOT perform loss aggregation, metric updates, logging, or autoregressive generation.

#### Scenario: Training step computes only teacher-forcing loss
- **WHEN** TIGER executes `training_step`
- **THEN** it MUST compute teacher-forcing decoder outputs and loss from `TigerModelInput` plus `TigerLabelData`
- **AND** it MUST NOT call autoregressive generation for that batch

#### Scenario: Evaluation step computes loss and generation explicitly
- **WHEN** TIGER executes validation or test evaluation for a labeled batch
- **THEN** it MUST compute teacher-forcing loss through the explicit loss path
- **AND** it MUST call autoregressive generation exactly once for evaluator metrics

#### Scenario: Prediction step produces generated outputs only
- **WHEN** TIGER executes `predict_step`
- **THEN** it MUST call autoregressive generation for the batch
- **AND** it MUST wrap generated semantic IDs in `ModelOutput`
- **AND** it MUST NOT return placeholder loss values

#### Scenario: Step methods defer mode switching to Lightning lifecycle
- **WHEN** maintainers inspect `training_step`, validation, test, and prediction step methods
- **THEN** those methods MUST NOT call `train()` or `eval()` to switch module mode
- **AND** mode-sensitive behavior MUST remain controlled by Lightning loops and model lifecycle hooks
