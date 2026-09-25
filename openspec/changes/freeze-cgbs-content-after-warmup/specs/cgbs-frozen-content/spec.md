## ADDED Requirements
### Requirement: Freeze complete content path
The model SHALL freeze encoder, shared SID embeddings, attention and query MLP after warmup while allowing decoder-only parameters and mixture logits to update.
#### Scenario: Momentum and shared parameters
- **WHEN** Adam has accumulated warmup state and joint training starts
- **THEN** frozen parameters and their optimizer step counters remain unchanged even if the decoder uses the shared embedding
### Requirement: Deterministic frozen representation
The frozen content path SHALL disable dropout and remain deterministic across training-mode resets.
#### Scenario: Return from validation
- **WHEN** the model returns to train mode after validation
- **THEN** identical history produces identical frozen content queries
### Requirement: Resume identity
Frozen-content training SHALL record its schedule and restore freeze behavior from global_step without changing inference architecture.
#### Scenario: Mismatched training resume
- **WHEN** a non-frozen training checkpoint is supplied for resume
- **THEN** the frozen variant rejects the schedule mismatch
