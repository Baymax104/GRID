# tail-candidate-allocation-probe-analysis Specification

## Purpose
TBD - created by archiving change add-prefix-balanced-candidate-allocation-probe. Update Purpose after archive.
## Requirements
### Requirement: Probe analysis SHALL compare matched baseline and intervention runs
Candidate allocation analysis SHALL require baseline and intervention recommendation outputs and Prefix Trace inputs with exactly matched keys, labels, checkpoint, semantic ID lineage, data split, beam width, seed, and SID shape. Baseline allocation MUST be disabled and intervention allocation MUST be enabled.

#### Scenario: Matched pair is supplied
- **WHEN** all pair identity fields and row-level keys and labels match
- **THEN** analysis MUST proceed using paired per-user transitions

#### Scenario: Pair identity differs
- **WHEN** any required identity field differs or lineage is ambiguous
- **THEN** analysis MUST fail before emitting a verdict
- **AND** the mismatch MUST be named in the error or audit output

### Requirement: Probe analysis SHALL separate candidate access from Top10 outcome
The analysis SHALL report baseline/intervention candidate reach, Top10 hits, additions, losses, net changes, and prefix survival separately for Overall, Head, Mid, Tail, and Tail-Cold groups when supported.

#### Scenario: Target becomes reachable but remains outside Top10
- **WHEN** an intervention target enters the generated candidate set but is not a Top10 hit under original model score
- **THEN** it MUST count as candidate-access recovery
- **AND** it MUST NOT count as a recommendation hit improvement

#### Scenario: Intervention both adds and loses hits
- **WHEN** paired users include baseline misses becoming hits and baseline hits becoming misses
- **THEN** evidence MUST report both counts and their net change

### Requirement: Probe evidence SHALL expose allocation and survival mechanisms
Structured evidence SHALL include target shortlist membership, reserve retention, prefix training mass, first failure depth, and per-layer survival deltas, joined by audited user key and frequency group.

#### Scenario: Structured evidence is written
- **WHEN** a valid probe pair is analyzed
- **THEN** evidence MUST include per-user and grouped allocation outcome tables
- **AND** it MUST include per-layer prefix survival and reserve-use tables
- **AND** manifest metadata MUST preserve every input reference and resolved Artifact identity

### Requirement: Probe verdict SHALL enforce recommendation-level gates
The verdict SHALL use predeclared gates and MUST NOT advance solely from improved prefix survival or candidate reach. `advance` SHALL require Tail candidate reach to improve in at least three of four registered settings, real Tail Top10 additions in at least two settings, overall Hit@10 loss no worse than 0.2 percentage points, and Head Hit@10 loss no worse than 0.5 percentage points.

#### Scenario: Only intermediate metrics improve
- **WHEN** Tail survival or candidate reach improves but no registered setting adds a Tail Top10 hit
- **THEN** the verdict MUST NOT be `advance`

#### Scenario: Cost guardrail is violated
- **WHEN** overall or Head Hit@10 loss exceeds its configured guardrail
- **THEN** the affected setting MUST fail the probe gate regardless of Tail intermediate improvements

#### Scenario: Evidence is incomplete
- **WHEN** fewer than four registered settings are available for the cross-setting verdict
- **THEN** the cross-setting verdict MUST be `inconclusive`
- **AND** per-setting evidence MUST remain available without being promoted to a general conclusion

### Requirement: Complete probe execution SHALL remain manual
Repository automation SHALL provide commands and dry-run validation but MUST NOT automatically execute full inference, diagnosis, or online publication for the registered dataset and quantizer settings.

#### Scenario: Implementation validation runs
- **WHEN** maintainers execute the change's automated verification tasks
- **THEN** those tasks MUST be limited to unit tests, config composition, launcher argument checks, syntax checks, and OpenSpec validation
- **AND** full experiment commands MUST remain documented manual actions
