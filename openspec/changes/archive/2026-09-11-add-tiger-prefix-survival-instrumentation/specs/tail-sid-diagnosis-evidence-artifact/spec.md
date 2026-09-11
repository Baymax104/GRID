## ADDED Requirements

### Requirement: Tail-SID diagnosis SHALL optionally consume Prefix Trace Artifacts
Tail-SID diagnosis SHALL accept an optional fixed-beam Prefix Trace Artifact and an optional widened-beam Prefix Trace Artifact through explicit configuration fields and the shared Artifact resolver. Trace inputs SHALL be joined to recommendation labels and item evidence by business keys, not row positions.

#### Scenario: Fixed-beam trace is provided
- **WHEN** diagnosis receives a valid fixed-beam Prefix Trace Artifact
- **THEN** it MUST validate user keys, target labels, semantic-ID reference, data split and schema version
- **AND** it MUST compute layer-wise target probability/rank, survival and first-failure evidence

#### Scenario: Widened-beam trace is also provided
- **WHEN** diagnosis receives compatible fixed and widened trace Artifacts
- **THEN** it MUST validate checkpoint, split, labels and semantic-ID identity across them
- **AND** it MUST compute target-path recovery and failure-depth shift evidence

#### Scenario: Trace inputs are absent
- **WHEN** diagnosis runs without Prefix Trace Artifacts
- **THEN** all existing static and recommendation evidence behavior MUST remain available
- **AND** prefix-survival mechanism evidence MUST be marked unavailable

### Requirement: Diagnosis SHALL emit stable prefix-survival mechanism evidence
When trace evidence is available, diagnosis SHALL extend its structured output with stable layer/group survival, first-failure, competition-association and optional widened-beam recovery files plus a separately named mechanism verdict.

#### Scenario: Mechanism evidence is written
- **WHEN** fixed-beam trace analysis completes
- **THEN** the evidence directory MUST include layer-wise group metrics and first-failure metrics
- **AND** `summary.json` MUST report trace schema, split, beam width and mechanism-evidence availability

#### Scenario: Recovery evidence is written
- **WHEN** compatible widened-beam trace is available
- **THEN** the evidence directory MUST include widened-beam recovery metrics by popularity group and layer
- **AND** summary MUST distinguish recovery evidence from ordinary recommendation outcome metrics

### Requirement: Mechanism evidence SHALL preserve exploratory and confirmatory boundaries
Diagnosis SHALL record whether trace inputs came from evaluation or testing. Outputs intended as calibration statistics MUST reject testing input, and the H2 verdict SHALL remain distinct from existing structural and generation-risk verdicts.

#### Scenario: Evaluation trace is used for method development
- **WHEN** diagnosis is configured to emit calibration-statistics-ready evidence
- **THEN** every trace input MUST declare `data_split=evaluation`
- **AND** summary metadata MUST identify the output as development evidence

#### Scenario: Testing trace is presented as calibration statistics
- **WHEN** a testing trace is supplied to a calibration-statistics-ready analysis
- **THEN** diagnosis MUST fail with a data leakage error

#### Scenario: H2 evidence is unavailable or unsupported
- **WHEN** layer-wise survival or recovery evidence does not meet configured availability or stability rules
- **THEN** diagnosis MUST NOT change `raw_damage` or reuse `generation_risk_validity` as the H2 verdict
- **AND** the separately named mechanism verdict MUST report unavailable or not supported
