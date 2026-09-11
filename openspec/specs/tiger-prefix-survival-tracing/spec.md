# tiger-prefix-survival-tracing Specification

## Purpose
TBD - created by archiving change add-tiger-prefix-survival-instrumentation. Update Purpose after archive.
## Requirements
### Requirement: TIGER SHALL optionally emit target-centric teacher-forcing traces
Trace-enabled TIGER inference SHALL compute per-user, per-hierarchy target token probability, legal-candidate rank, and target-vs-best-legal margin from teacher-forcing logits using the labeled target SID. The trace MUST use the same hierarchy-local vocabulary and legal-prefix catalog as generation.

#### Scenario: Teacher-forcing trace is computed for a labeled batch
- **WHEN** trace-enabled inference receives `TigerModelInput` and `TigerLabelData`
- **THEN** it MUST emit target token probability, legal rank, and margin tensors with shape `(batch_size, num_hierarchies)`
- **AND** hierarchy values MUST correspond to the target model SID including the dedup hierarchy

#### Scenario: Trace inference has no target labels
- **WHEN** trace-enabled inference receives a batch without `TigerLabelData`
- **THEN** it MUST fail before publishing a Prefix Trace Artifact
- **AND** the error MUST identify that target labels are required for prefix survival tracing

### Requirement: TIGER SHALL optionally emit target-centric beam survival traces
Trace-enabled constrained beam search SHALL observe the ground-truth target path at every hierarchy without modifying legality masks, candidate scores, parent selection, or top-k results. It SHALL emit target prefix survival, beam rank, parent beam rank, target path score, cutoff score, cutoff margin, and legal candidate count.

#### Scenario: Target prefix survives a hierarchy
- **WHEN** the ground-truth prefix is present after top-k pruning at hierarchy `l`
- **THEN** `target_prefix_survived[l]` MUST be true
- **AND** its beam rank MUST be 1-based and identify its position in the retained beam

#### Scenario: Target prefix first exits the beam
- **WHEN** the ground-truth prefix is absent after top-k pruning for the first time at hierarchy `l`
- **THEN** `first_failure_depth` MUST equal the 1-based depth `l + 1`
- **AND** beam rank at that and subsequent absent layers MUST use sentinel `-1`

#### Scenario: Target path survives all hierarchies
- **WHEN** the ground-truth path remains in the beam through the final hierarchy
- **THEN** `first_failure_depth` MUST be `-1`

### Requirement: Prefix tracing SHALL preserve default generation behavior
Instrumentation SHALL be optional and SHALL NOT change generated semantic IDs, marginal path scores, legal-prefix enforcement, or standard recommendation output when disabled or enabled.

#### Scenario: Tracing is disabled
- **WHEN** TIGER inference runs with tracing disabled
- **THEN** decoder and model return contracts MUST remain compatible with existing inference
- **AND** standard recommendation output MUST remain unchanged

#### Scenario: Tracing is enabled on the same deterministic input
- **WHEN** the same model, batch, beam width, and random state are evaluated with tracing disabled and enabled
- **THEN** generated semantic IDs and marginal path scores MUST be element-wise identical

### Requirement: Widened-beam traces SHALL be comparable without retraining
The trace experiment SHALL allow runtime beam width override while loading the same TIGER checkpoint. Fixed and widened runs intended for recovery analysis MUST record sufficient identity metadata for exact comparison.

#### Scenario: Beam width is overridden at inference
- **WHEN** an existing checkpoint is loaded with a different `top_k_for_generation`
- **THEN** inference MUST use the configured runtime beam width without requiring checkpoint retraining
- **AND** the trace metadata MUST record that beam width

#### Scenario: Registered pilot widths are used
- **WHEN** the four-checkpoint mechanism pilot is executed
- **THEN** each setting MUST produce comparable K=10 and K=50 traces on the same evaluation keys
