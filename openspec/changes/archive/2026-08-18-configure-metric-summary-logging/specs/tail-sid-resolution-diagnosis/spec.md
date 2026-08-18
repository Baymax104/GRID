## ADDED Requirements

### Requirement: Tail SID diagnosis SHALL log metrics as run summaries
Tail-SID diagnosis SHALL configure framework-managed test metrics as run-level summary values so W&B records the final diagnosis scalar values without creating metric history curves.

#### Scenario: Diagnosis test metrics use summary mode
- **WHEN** `experiment=tail_sid_diagnosis` is composed
- **THEN** the metric callback configuration MUST set test metric logging to summary mode
- **AND** the diagnosis metric definitions MUST remain split across structural, semantic, damage, and prefix-risk metrics

#### Scenario: Diagnosis remains on Lightning test path
- **WHEN** Tail-SID diagnosis runs
- **THEN** it MUST continue to use the unified launcher and `Trainer.test`
- **AND** it MUST NOT introduce a separate diagnosis runner or diagnosis-specific logging callback
