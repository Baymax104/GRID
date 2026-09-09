# tail-sid-resolution-diagnosis Specification

## Purpose
TBD - created by archiving change add-tail-sid-diagnosis-core. Update Purpose after archive.
## Requirements
### Requirement: Tail SID diagnosis SHALL load keyed bundle inputs
The system SHALL provide an offline Tail-SID Resolution Damage diagnosis entrypoint that loads Semantic ID predictions from a keyed prediction bundle and optionally loads item embeddings from a keyed prediction bundle. The diagnosis SHALL NOT assume item ids are contiguous tensor row indexes. The official diagnosis DataModule MUST resolve local or W&B-backed bundle references with their actual field semantics and explicit experiment-provided W&B identity before constructing the diagnosis dataset.

#### Scenario: Semantic ID bundle is loaded by key
- **WHEN** the diagnosis runner receives `semantic_id_path`
- **THEN** it MUST load the file through the keyed prediction bundle loader
- **AND** it MUST preserve the item keys associated with each SID row

#### Scenario: Short Semantic ID URI uses diagnosis experiment identity
- **WHEN** the official diagnosis receives `semantic_id_path=wandb://<run-id>`
- **AND** the experiment config provides `user/project`
- **THEN** the diagnosis DataModule MUST resolve the reference with `field_name="semantic_id_path"`
- **AND** it MUST pass the experiment user and project as explicit resolver defaults
- **AND** the selected Artifact role MUST be `semantic_id`
- **AND** the diagnosis dataset MUST receive the resolved local bundle path

#### Scenario: Optional embedding URI uses embedding field semantics
- **WHEN** the official diagnosis receives a non-null `embedding_path=wandb://<run-id>`
- **AND** the experiment config provides `user/project`
- **THEN** the diagnosis DataModule MUST resolve the reference with `field_name="embedding_path"`
- **AND** the selected Artifact role MUST be `semantic_embedding`
- **AND** the diagnosis dataset MUST receive the resolved local bundle path

#### Scenario: Missing optional embedding bypasses resolution
- **WHEN** the official diagnosis config sets `embedding_path: null`
- **THEN** the diagnosis DataModule MUST NOT resolve or download an embedding Artifact
- **AND** diagnosis metric computation MUST continue without embeddings

#### Scenario: Local bundle paths remain compatible
- **WHEN** diagnosis input references are local paths
- **THEN** reference resolution MUST return the paths without initializing W&B API
- **AND** the diagnosis dataset MUST continue loading the existing keyed bundle files

#### Scenario: Raw and model SID views are retained
- **WHEN** the SID prediction width is greater than `raw_num_hierarchies`
- **THEN** the diagnosis MUST treat the leading `raw_num_hierarchies` columns as `raw_sid`
- **AND** it MUST retain the full prediction row as `model_sid`
- **AND** it MUST expose the final extra column as a deduplication digit

### Requirement: Tail SID diagnosis SHALL derive frequency groups from training data
The diagnosis SHALL scan `data_dir/training` sequence records to compute training frequency per item and assign each known item to Head, Mid, Tail, or Tail-Cold groups without using evaluation or testing labels for the primary split. This frequency grouping SHALL be expressed as data preprocessing for the diagnosis dataset.

#### Scenario: Training sequence frequencies are grouped
- **WHEN** the diagnosis dataset loads semantic IDs and training sequence records
- **THEN** it MUST compute `freq_train` from `sequence_data`
- **AND** a configured preprocessing function MUST assign non-cold items by configured head and tail ratios
- **AND** it MUST assign known items with zero training frequency to `Tail-Cold`

### Requirement: Tail SID diagnosis SHALL compute structural damage metrics
The diagnosis SHALL compute structural SID information from raw SID prefix buckets as part of configured diagnosis metrics. The official analysis experiment SHALL expose loggable summary metrics through the shared metric runtime rather than writing item-level structural rows.

#### Scenario: Collision and near-collision summary metrics are recorded
- **WHEN** Tail-SID diagnosis runs as an analysis experiment
- **THEN** configured metric computation MUST account for full collisions and strict near-collisions
- **AND** the standard metric callback MUST record scalar summary values derived from those computations

#### Scenario: Prefix density summary metrics are recorded
- **WHEN** multiple items share raw SID prefixes
- **THEN** configured metric computation MUST account for prefix density
- **AND** the standard metric callback MUST record scalar summary values derived from density computations

#### Scenario: Suffix burden contributes to summary damage
- **WHEN** multiple items share the first `L-1` raw SID tokens without full collision
- **THEN** configured metric computation MUST account for last-step burden when producing scalar damage summary metrics

### Requirement: Tail SID diagnosis SHALL compute semantic mismatch when embeddings are available
The diagnosis SHALL compute semantic mismatch for strict deep-overlap neighbors when `embedding_path` is provided. It SHALL skip semantic mismatch with zero-valued contributions when embeddings are not provided. The official analysis experiment SHALL expose semantic mismatch through loggable summary metrics.

#### Scenario: Semantic mismatch uses item embeddings
- **WHEN** the diagnosis receives `embedding_path`
- **THEN** configured metric computation MUST compute cosine similarity for sampled strict deep-overlap neighbors
- **AND** the standard metric callback MUST record scalar semantic mismatch summary values

#### Scenario: Semantic mismatch is optional
- **WHEN** the diagnosis does not receive `embedding_path`
- **THEN** the diagnosis metric computation MUST still complete
- **AND** semantic mismatch contributions MUST be zero-valued in recorded summary metrics

### Requirement: Tail SID diagnosis SHALL be runnable from the project root
The project SHALL provide a root-level script for running the diagnosis from the repository root through the unified Hydra entrypoint.

#### Scenario: Root script documents required arguments
- **WHEN** a maintainer opens the root diagnosis script
- **THEN** it MUST call `uv run --module src.main experiment=tail_sid_diagnosis`
- **AND** it MUST expose `data_dir`, `semantic_id_path`, and `raw_num_hierarchies` as editable overrides

### Requirement: Tail SID diagnosis SHALL use stable damage normalization
Tail-SID diagnosis SHALL compute composite damage scores with a stable normalization strategy that does not amplify near-constant metric components into extremely large scores. The official analysis experiment SHALL record scalar normalization and damage summary metrics through the shared metric runtime.

#### Scenario: Degenerate metric component is neutralized
- **WHEN** a metric component has an interquartile range below the configured stability threshold
- **THEN** that component MUST contribute zero to composite `damage`
- **AND** the system MUST NOT divide by a tiny epsilon in a way that produces unbounded scores

#### Scenario: Metric contribution is bounded
- **WHEN** a metric component has non-degenerate spread
- **THEN** its normalized contribution MUST be clamped to a finite configured range before being added to `damage`

#### Scenario: Score metadata is recorded as metrics
- **WHEN** diagnosis metrics are computed
- **THEN** scalar score normalization metadata MUST be available as logged metrics where numeric
- **AND** non-numeric explanatory metadata MUST NOT be sent to Lightning `log_dict`

### Requirement: Tail SID diagnosis SHALL remove the public compute_metrics entrypoint
Tail-SID diagnosis SHALL use `TailSIDDiagnosisMetric` as the public metric computation entrypoint. The module SHALL NOT expose `compute_metrics(...)` as a public function.

#### Scenario: Package exports structured metric entrypoint
- **WHEN** a maintainer imports public Tail-SID diagnosis symbols
- **THEN** `TailSIDDiagnosisMetric` MUST be exported
- **AND** `compute_metrics` MUST NOT be exported
- **AND** `run_diagnosis` MUST NOT be exported
- **AND** `DiagnosisConfig` MUST NOT be exported

#### Scenario: Existing formulas and output schemas are preserved
- **WHEN** diagnosis metrics are computed through `TailSIDDiagnosisMetric`
- **THEN** item-level collision, density, suffix burden, semantic mismatch, damage, group rows, prefix rows, and summary fields MUST preserve their existing meanings
- **AND** reporting output file schemas MUST remain unchanged

### Requirement: Tail SID diagnosis SHALL run as an official analysis experiment
Tail-SID diagnosis SHALL be launched through the unified Hydra main entrypoint as a `run_mode: analysis` experiment and SHALL load its test inputs through the shared diagnosis DataModule/Dataset path.

#### Scenario: Diagnosis experiment declares data-layer diagnosis datamodule
- **WHEN** a maintainer composes `experiment=tail_sid_diagnosis`
- **THEN** the data datamodule target MUST be `src.data.datamodule.DiagnosisDataModule`
- **AND** the test dataset target MUST be the shared diagnosis dataset
- **AND** the quantization package MUST NOT define a Tail-SID-specific DataModule for official runs

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

