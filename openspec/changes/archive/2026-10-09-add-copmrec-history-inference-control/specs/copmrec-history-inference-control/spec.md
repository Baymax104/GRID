## ADDED Requirements

### Requirement: Fixed checkpoint final history mask control

The system SHALL provide a single-process inference control that restores the formal CoPMRec Full checkpoint and preserves history encoding, residual parameters, full-catalog logits and stable catalog-row ordering while omitting only the final history exclusion mask. It SHALL forbid training and checkpoint replacement.

#### Scenario: Authorized Beauty seed42 inference

- **WHEN** the user authorizes one Testing pass using the validation-selected Full checkpoint
- **THEN** the system runs through src.main and the shared output writers on one GPU and records immutable checkpoint, input and runtime source identities

#### Scenario: Independent empirical verification

- **WHEN** the prediction bundle is complete
- **THEN** the system checks matching users and labels, legal unique predictions, permits history overlap, reports target-in-history counts and independently recomputes metrics and paired differences without new model forwards
