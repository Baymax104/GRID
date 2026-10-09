## ADDED Requirements
### Requirement: Masked attention pooling
Content queries SHALL optionally use learned attention weights over valid encoder positions, preserving default mean pooling.
#### Scenario: Padding and initialization
- **WHEN** attention pooling is initialized
- **THEN** weights are uniform on valid positions and padding cannot affect the query
#### Scenario: Learning
- **WHEN** content loss is optimized during warmup
- **THEN** attention parameters receive gradients while decoder-exclusive parameters remain unchanged
### Requirement: Model identity
Attention pooling SHALL be encoded in the checkpoint catalog contract and used identically in training and inference.
#### Scenario: Wrong model restoration
- **WHEN** an attention checkpoint is loaded into mean pooling or conversely
- **THEN** restoration fails explicitly
### Requirement: Frozen combined trial
The new configuration SHALL use hidden width512, warmup10000, total30000 updates and existing validation content loss.
#### Scenario: Stage switch
- **WHEN** 10000 optimizer updates have completed
- **THEN** subsequent updates use the original joint objective and existing Adam state
