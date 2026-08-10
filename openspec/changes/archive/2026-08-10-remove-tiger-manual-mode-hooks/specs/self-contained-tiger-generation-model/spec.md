## ADDED Requirements

### Requirement: TIGER SHALL rely on Lightning lifecycle for train and eval mode
TIGER SHALL NOT define custom lifecycle hooks whose only responsibility is manually switching wrapped encoder or decoder modules between train and eval mode. Standard training, validation, test, and prediction loops SHALL rely on Lightning lifecycle behavior for module mode management.

#### Scenario: Manual mode hooks are absent
- **WHEN** maintainers inspect the TIGER LightningModule
- **THEN** it MUST NOT define helper methods that manually set wrapped encoder or decoder train/eval mode for Lightning loops
- **AND** it MUST NOT set custom `is_training` attributes on wrapped encoder or decoder modules

#### Scenario: Metric lifecycle hooks remain
- **WHEN** TIGER starts train, validation epoch, or test epoch lifecycle events
- **THEN** metric reset behavior MUST remain available
- **AND** those hooks MUST NOT perform manual train/eval mode switching

#### Scenario: Step methods remain mode-neutral
- **WHEN** maintainers inspect TIGER `training_step`, validation, test, and prediction step methods
- **THEN** those methods MUST NOT call `train()` or `eval()` directly
