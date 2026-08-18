# tail-sid-resolution-diagnosis Specification

## Purpose
TBD - created by archiving change add-tail-sid-diagnosis-core. Update Purpose after archive.
## Requirements
### Requirement: Tail SID diagnosis SHALL load keyed bundle inputs
The system SHALL provide an offline Tail-SID Resolution Damage diagnosis entrypoint that loads Semantic ID predictions from a keyed prediction bundle and optionally loads item embeddings from a keyed prediction bundle. The diagnosis SHALL NOT assume item ids are contiguous tensor row indexes.

#### Scenario: Semantic ID bundle is loaded by key
- **WHEN** the diagnosis runner receives `semantic_id_path`
- **THEN** it MUST load the file through the keyed prediction bundle loader
- **AND** it MUST preserve the item keys associated with each SID row

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

