## ADDED Requirements

### Requirement: Frozen candidate replay
The diagnosis SHALL score original on/off candidates using matched token and mixed path scores without training or new beam search.

#### Scenario: Matched inputs
- **WHEN** checkpoint, traces, labels and candidates match
- **THEN** four candidate-pool/score cells are exported with provenance

### Requirement: Reproduction gate
The diagnosis SHALL reject incompatible input lineage and diagonal ranking reproduction failures.

#### Scenario: Invalid reproduction
- **WHEN** the reference score cannot reproduce original candidate ordering within the fixed tie policy
- **THEN** the diagnosis fails before evidence is published

#### Scenario: Explicit reproduction audit
- **WHEN** reproduction_audit is explicitly enabled
- **THEN** diagonal ordering failures are collected across batches into a local report with score inversions, original and rescored target ranks, and NDCG changes
- **AND** source lineage, shape, score and metric validation remain active
- **AND** incomplete coverage or any diagonal failure produces a failed final verdict and no scientific evidence artifact is published

### Requirement: Manual bounded execution
The launcher SHALL use the unified experiment entrypoint, support dry-run, notes and overrides, and prohibit testing.

#### Scenario: User launch
- **WHEN** the user starts the evaluation diagnosis
- **THEN** only frozen candidate rescoring runs and a keyed artifact is written
