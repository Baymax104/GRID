## ADDED Requirements

### Requirement: Fixed single max-only admission
The analyzer SHALL construct Top10 by retaining all mass20 and cold candidates while admitting at most the highest-content-scoring max-only incremental candidate.

#### Scenario: Max-only candidates compete for one slot
- **WHEN** multiple candidates occur in max20 but not mass20 or cold
- **THEN** only the highest-content-scoring such candidate remains eligible before stable content ranking

#### Scenario: Cold candidates are common candidates
- **WHEN** a cold candidate also occurs in max20
- **THEN** it SHALL not consume the single max-only incremental slot

### Requirement: Frozen held-out decision
The analyzer SHALL use the existing key-XOR-seed42 selection/audit split and SHALL not scan quota values.

#### Scenario: Selection gate fails
- **WHEN** q=1 does not improve NDCG point estimates over both mass20 and mass30 with nondecreasing Recall
- **THEN** the analyzer SHALL stop without computing audit comparisons

#### Scenario: Audit gate passes
- **WHEN** selection passes and both audit NDCG confidence-interval lower bounds are positive with nondecreasing Recall
- **THEN** the analyzer SHALL retain only an evaluation candidate requiring independent confirmation

### Requirement: Existing-cache-only validation
The validation SHALL consume a valid `liger_source_protection_v1` cache and SHALL require no training, decoder search, or testing data.

#### Scenario: Cache replay
- **WHEN** the analysis runs
- **THEN** it validates the cache and reports mass20, union, mass30, and q=1 metrics with paired comparisons
