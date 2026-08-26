## ADDED Requirements

### Requirement: Tail SID diagnosis SHALL separate observed damage from frequency priority
The diagnosis SHALL compute an ungated `raw_damage` from observed structural and semantic components and SHALL keep any frequency-aware prioritization in a separately named `priority_score`. Evidence summaries, effect sizes, confidence intervals, correlations, and verdicts used to evaluate tail-specific damage MUST use raw components or `raw_damage` and MUST NOT use the frequency priority multiplier.

#### Scenario: Tail priority cannot prove tail damage
- **WHEN** the diagnosis compares Head, Mid, Tail, or Tail-Cold evidence
- **THEN** it MUST report ungated raw component values and `raw_damage`
- **AND** it MUST NOT use `priority_score` to decide whether Tail damage exceeds Head damage

#### Scenario: Priority score remains available for downstream consumers
- **WHEN** frequency-aware prioritization is enabled
- **THEN** the diagnosis MUST compute `priority_score` separately from `raw_damage`
- **AND** output metadata MUST record the configured group multipliers

### Requirement: Tail SID diagnosis SHALL report group-comparable raw evidence
The diagnosis SHALL report the same raw structural, semantic, frequency-asymmetric, and damage distribution fields for Head, Mid, Tail, and Tail-Cold whenever a group has members. Group summaries SHALL include group size so empty or low-support groups are distinguishable from zero risk.

#### Scenario: All frequency groups receive the same component schema
- **WHEN** diagnosis computes group evidence
- **THEN** Head, Mid, Tail, and Tail-Cold rows MUST use the same metric columns
- **AND** those columns MUST include collision, strict near-collision, local density, suffix weakness, last-step burden, semantic mismatch, harmful overlap, and `raw_damage`

#### Scenario: Damage distributions are not reduced to means only
- **WHEN** a non-empty frequency group is summarized
- **THEN** the diagnosis MUST report its mean and configured distribution quantiles for `raw_damage`
- **AND** it MUST report `num_items` and average training frequency for interpretation

### Requirement: Tail SID diagnosis SHALL quantify frequency-asymmetric overlap
The diagnosis SHALL distinguish Tail-Head, Tail-Mid, Tail-Tail, and Tail-Cold overlap relationships for strict deep-prefix neighborhoods and full-collision buckets. It SHALL expose head-dominated bucket membership, Tail isolation deficit, and Tail-to-Head deep-overlap pressure as raw evidence components.

#### Scenario: Tail overlap partners are typed by frequency group
- **WHEN** a Tail or Tail-Cold item shares a strict deep prefix with other items
- **THEN** the diagnosis MUST count overlap neighbors separately by partner group
- **AND** full collisions MUST remain distinguishable from strict near-collisions

#### Scenario: Head-dominated buckets are identified
- **WHEN** a collision or deep-prefix bucket contains Tail and non-Tail items
- **THEN** the diagnosis MUST report the bucket group composition
- **AND** it MUST derive head-dominance and Tail isolation evidence without applying a Tail priority gate

### Requirement: Tail SID diagnosis SHALL distinguish global semantic mismatch from bucket-relative outliers
When embeddings are provided, the diagnosis SHALL compute a primary global semantic mismatch against a deterministic random-pair similarity reference distribution and MAY additionally compute a bucket-relative semantic outlier score. The two fields MUST have distinct names and metadata and MUST NOT be combined as if they shared the same threshold semantics.

#### Scenario: Global random-pair threshold is reproducible
- **WHEN** global semantic mismatch is enabled
- **THEN** the diagnosis MUST sample random item pairs with a configured seed and bounded sample count
- **AND** it MUST record the sampled-pair count, similarity quantile, threshold, and seed

#### Scenario: Bucket-relative sensitivity result is separately labeled
- **WHEN** bucket-relative semantic analysis is enabled
- **THEN** its threshold MUST be computed within the applicable strict-prefix bucket
- **AND** its output MUST be named and reported separately from global semantic mismatch

#### Scenario: Missing embeddings preserve structural diagnosis
- **WHEN** `embedding_path` is null
- **THEN** structural, frequency-asymmetric, and raw damage outputs MUST still be produced
- **AND** semantic fields and verdict support MUST explicitly indicate that semantic evidence is unavailable rather than treating missing evidence as observed agreement

### Requirement: Tail SID diagnosis SHALL optionally relate damage to recommendation outcomes
The official diagnosis SHALL accept an optional `recommendation_output_path` keyed TIGER inference bundle. When provided, it SHALL align recommendations to testing labels by user key, derive item-level hit/rank outcomes, and evaluate whether raw damage predicts recommendation failures overall and within Tail groups.

#### Scenario: Recommendation bundle is resolved with explicit identity and lineage
- **WHEN** `recommendation_output_path` is a local or W&B-backed reference
- **THEN** the diagnosis DataModule MUST resolve it with field semantics for recommendation output and explicit experiment identity
- **AND** a W&B-backed reference MUST be available to the existing lineage callback before test execution

#### Scenario: User-key alignment is required
- **WHEN** recommendation correlation is computed
- **THEN** testing labels and generated SID candidates MUST be joined by user key
- **AND** the diagnosis MUST NOT assume bundle row order matches testing-record order

#### Scenario: Item outcomes are aggregated from user-level predictions
- **WHEN** a testing label and generated candidate list are aligned
- **THEN** the diagnosis MUST compute configured hit@K, rank, and NDCG contribution values
- **AND** it MUST aggregate label count and outcome values by label item before item-level risk analysis

#### Scenario: Recommendation input remains optional
- **WHEN** `recommendation_output_path` is null
- **THEN** pure SID diagnosis MUST continue to complete
- **AND** recommendation-correlation fields and verdict support MUST explicitly indicate that outcome evidence was not evaluated

### Requirement: Tail SID diagnosis SHALL provide reproducible evidence tests and verdicts
The diagnosis SHALL report Tail-versus-Head absolute differences, ratios where defined, deterministic bootstrap confidence intervals, frequency-matched outcome comparisons, and configured sensitivity results. It SHALL produce a machine-readable verdict that distinguishes structural asymmetry, equal-risk Tail vulnerability, generation-risk validity, and insufficient evidence.

#### Scenario: Tail versus Head effect size is reported
- **WHEN** both Tail and Head contain observations for a raw component
- **THEN** the diagnosis MUST report the configured difference and ratio statistics
- **AND** it MUST report a deterministic bootstrap confidence interval without using `priority_score`

#### Scenario: Frequency is controlled in recommendation analysis
- **WHEN** recommendation outcomes are available
- **THEN** the diagnosis MUST compare low- and high-damage items within configured training-frequency bins
- **AND** it MUST report bin support so sparse comparisons are not presented as conclusive evidence

#### Scenario: Sensitivity settings are explicit
- **WHEN** Tail-ratio, damage-component, or semantic-threshold sensitivity is requested
- **THEN** each result MUST be labeled with its complete setting
- **AND** the primary configured setting MUST remain distinguishable from sensitivity settings

#### Scenario: Verdict follows declared go-no-go rules
- **WHEN** diagnosis evidence is finalized
- **THEN** the machine-readable verdict MUST state which evidence dimensions passed, failed, or were unavailable
- **AND** it MUST NOT report tail-specific support solely because a frequency priority multiplier raised Tail scores
