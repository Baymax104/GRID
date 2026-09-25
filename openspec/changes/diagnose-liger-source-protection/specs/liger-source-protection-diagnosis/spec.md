## ADDED Requirements

### Requirement: Evaluation-only three-search cache
The system SHALL run only on the evaluation split and SHALL cache mass20, max20, and mass30 candidates from one shared encoder/content computation.

#### Scenario: Cache fixed candidate pools
- **WHEN** the diagnosis predicts an evaluation batch
- **THEN** it performs exactly three beam searches and records union-plus-cold and mass30-plus-cold pools with content scores

#### Scenario: Reject testing or training
- **WHEN** the diagnosis is configured for testing data or a training loss
- **THEN** the system fails closed

### Requirement: Exact replay and source membership
The cache SHALL record enough information to exactly replay beta0 union ranking, mass30 ranking, and mass20 source membership.

#### Scenario: Validate cached pools
- **WHEN** a cache shard or merged cache is validated
- **THEN** union rows equal the stable mass20/max20 union plus cold rows, mass30 rows equal mass30 plus cold rows, padding/scores are consistent, and source membership matches mass20

### Requirement: Frozen protection scan
The analyzer SHALL evaluate only beta values `[0, 0.1, 0.25, 0.5, 1.0]` using per-user standardized content scores plus a mass-source bonus.

#### Scenario: Replay beta zero
- **WHEN** beta is zero
- **THEN** the resulting Top10 SHALL equal the unmodified content ranking

#### Scenario: Select a stable plateau
- **WHEN** adjacent nonzero beta values both improve selection NDCG over beta0 and mass30 with nondecreasing Recall
- **THEN** the analyzer freezes the smaller beta for audit evaluation

### Requirement: Held-out audit decision
The analyzer SHALL use a deterministic 50/50 user split and SHALL evaluate the frozen beta on the audit half without changing it.

#### Scenario: No selection plateau
- **WHEN** no adjacent beta values qualify on selection
- **THEN** the analyzer stops without promoting a setting

#### Scenario: Audit fails
- **WHEN** a frozen beta does not have positive NDCG CI lower bounds against beta0 and mass30 or has negative Recall point delta on audit
- **THEN** the analyzer stops the source-protection route

#### Scenario: Audit passes
- **WHEN** both audit NDCG CI lower bounds are positive and both Recall point deltas are nonnegative
- **THEN** the analyzer retains an evaluation-only candidate requiring independent confirmation
