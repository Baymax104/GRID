## ADDED Requirements

### Requirement: Content-only warmup and joint training
The model SHALL use unweighted content CE before the configured optimizer-step boundary and the unchanged CGBS joint loss afterwards, with one persistent optimizer.

#### Scenario: Parameter updates
- **WHEN** a warmup update is performed
- **THEN** encoder, shared SID embeddings and query may update while decoder-only weights and mixture weights remain unchanged

#### Scenario: Boundary transition
- **WHEN** global_step reaches warmup_steps
- **THEN** the next update uses the original joint objective and decoder-only parameters receive gradients

### Requirement: Resume and selection
The schedule SHALL be stored in checkpoints and validated for training resume, and best model selection SHALL exclude warmup validation.

#### Scenario: Resume
- **WHEN** the schedule differs from the training checkpoint
- **THEN** training rejects it rather than silently restarting warmup

#### Scenario: Warmup validation
- **WHEN** validation completes at or before warmup_steps
- **THEN** it does not save a best checkpoint

### Requirement: Manual reproducible execution
The experiment SHALL retain the unified entrypoint, dry-run, notes and Hydra overrides and use DDP with unused parameter detection.

#### Scenario: Fixed-budget launch
- **WHEN** the user launches the frozen configuration
- **THEN** 5000 warmup updates precede 35000 joint updates with seed42 and the existing 128-dimensional structure

### Requirement: Validation content loss
The staged model SHALL record unweighted full-catalog content CE as val/content_loss in both phases, aggregated by user count, without changing val/loss or checkpoint selection.

#### Scenario: Unequal validation batch sizes
- **WHEN** validation batches contain different numbers of users
- **THEN** content CE is averaged across users rather than equally across batch means
