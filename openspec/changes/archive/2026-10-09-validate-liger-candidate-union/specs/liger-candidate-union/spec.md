## ADDED Requirements

### Requirement: Fixed candidate construction arms
The system SHALL expose exactly the inference-only arms `mass_max_union20` and `mass30` for the frozen joint checkpoint. The union arm SHALL construct a stable deduplicated union of mass20 and max20 generated rows; the control SHALL generate mass30 rows only.

#### Scenario: Construct union candidates
- **WHEN** `mass_max_union20` inference receives shared encoded states and content logits
- **THEN** the system performs mass beam20 and max beam20 and returns their per-user deduplicated union padded to width40

#### Scenario: Construct expanded mass control
- **WHEN** `mass30` inference receives shared encoded states and content logits
- **THEN** the system performs one mass beam30 search and returns those rows without a max search

#### Scenario: Reject training
- **WHEN** either candidate-union arm is used for a training loss
- **THEN** the system fails closed as an inference-only intervention

### Requirement: Preserve downstream ranking controls
The system SHALL preserve the joint checkpoint learned alpha, legal token support, cold-item union, shared content logits, stable dense reranking, and TopK output for both arms.

#### Scenario: Rank the generated pool
- **WHEN** an arm has produced its generated rows
- **THEN** the system unions valid generated rows with cold rows and ranks the pool by the unchanged content logits

### Requirement: Auditable union provenance
The system SHALL emit a keyed tensor trace that records source rows, output rows, target memberships, arm identity, beam widths, and search count. Validation SHALL reject any union that is not the exact stable deduplicated source union or whose target membership is not the logical OR of its sources.

#### Scenario: Validate an exact union
- **WHEN** a `mass_max_union20` trace is merged
- **THEN** every output row equals the stable deduplicated mass/max source union and target-output membership equals target-mass OR target-max

#### Scenario: Reject a fabricated union
- **WHEN** a source candidate is missing from the union output or an unrelated candidate is added
- **THEN** trace validation fails

### Requirement: Frozen three-arm decision
The system SHALL compare union20 with frozen mass20 and new mass30 by aligned user-level NDCG@10 and Recall@10, while checking shared catalog, labels, dense references, joint protocol, and runtime alpha. It SHALL issue one of three frozen decisions without parameter search.

#### Scenario: No conversion over mass20
- **WHEN** union-minus-mass20 NDCG@10 CI lower bound is non-positive or Recall@10 point delta is negative
- **THEN** the decision stops the candidate-union conversion route

#### Scenario: Budget-only effect
- **WHEN** union passes the mass20 gate but fails the same gate versus mass30
- **THEN** the decision records a candidate-budget effect without a unique complementarity claim

#### Scenario: Retain an exploratory complementarity candidate
- **WHEN** union passes both NDCG@10 CI and Recall@10 point gates
- **THEN** the decision retains a testing-informed candidate that requires independent confirmation
