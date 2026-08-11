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
The diagnosis SHALL scan `data_dir/training` sequence records to compute training frequency per item and assign each known item to Head, Mid, Tail, or Tail-Cold groups without using evaluation or testing labels for the primary split.

#### Scenario: Training sequence frequencies are grouped
- **WHEN** the diagnosis receives a data directory with training sequence records
- **THEN** it MUST compute `freq_train` from `sequence_data`
- **AND** it MUST assign non-cold items by configured head and tail ratios
- **AND** it MUST assign known items with zero training frequency to `Tail-Cold`

#### Scenario: Missing training records fail clearly
- **WHEN** the diagnosis cannot read any training sequence records
- **THEN** it MUST fail with a clear error that names the missing or unreadable training input

### Requirement: Tail SID diagnosis SHALL compute structural damage metrics
The diagnosis SHALL compute item-level structural SID metrics from raw SID prefix buckets, including full collision, maximum prefix overlap depth, strict near-collision count, local density, suffix weakness, and last-step burden.

#### Scenario: Collision and near-collision metrics are emitted
- **WHEN** two or more items share identical raw SID
- **THEN** each item in that raw SID bucket MUST have `full_collision_flag=1`
- **AND** `full_collision_size` MUST equal the bucket size

#### Scenario: Prefix density metrics are emitted
- **WHEN** multiple items share a raw SID prefix
- **THEN** each item in that prefix bucket MUST receive depth-specific density values
- **AND** `local_density` MUST increase with deeper crowded prefix buckets

#### Scenario: Suffix burden metrics are emitted
- **WHEN** multiple items share the first `L-1` raw SID tokens without full collision
- **THEN** affected items MUST receive positive `last_step_burden`

### Requirement: Tail SID diagnosis SHALL compute semantic mismatch when embeddings are available
The diagnosis SHALL compute semantic mismatch for strict deep-overlap neighbors when `--embedding-path` is provided. It SHALL skip semantic mismatch with zero-valued fields when embeddings are not provided.

#### Scenario: Semantic mismatch uses item embeddings
- **WHEN** the diagnosis receives `--embedding-path`
- **THEN** it MUST compute cosine similarity for sampled strict deep-overlap neighbors
- **AND** it MUST emit item-level `semantic_mismatch`
- **AND** it MUST emit item-level `qualified_harmful_overlap_count`

#### Scenario: Semantic mismatch is optional
- **WHEN** the diagnosis does not receive `--embedding-path`
- **THEN** it MUST still emit all required output files
- **AND** semantic mismatch fields MUST be present with zero values

### Requirement: Tail SID diagnosis SHALL emit reusable diagnosis outputs
The diagnosis SHALL create a clear output directory containing summary, group metrics, item damage scores, and prefix risk scores. The analysis runner SHALL also print a concise human-readable run summary.

#### Scenario: Required output files are written
- **WHEN** the diagnosis completes successfully
- **THEN** the output directory MUST contain `summary.json`
- **AND** it MUST contain `group_metrics.csv`
- **AND** it MUST contain `item_damage_scores.csv`
- **AND** it MUST contain `prefix_risk_scores.csv`

#### Scenario: Summary displays key group metrics
- **WHEN** the diagnosis completes successfully
- **THEN** stdout MUST include the output directory
- **AND** stdout MUST include Head, Mid, Tail group rows when those groups exist
- **AND** `summary.json` MUST include at least tail collision, tail near-collision, tail density, and average group damage fields

### Requirement: Tail SID diagnosis SHALL be runnable from the project root
The project SHALL provide a root-level script for running the diagnosis from the repository root through the unified Hydra entrypoint.

#### Scenario: Root script documents required arguments
- **WHEN** a maintainer opens the root diagnosis script
- **THEN** it MUST call `uv run --module src.main experiment=tail_sid_diagnosis`
- **AND** it MUST expose `data_dir`, `semantic_id_path`, and `raw_num_hierarchies` as editable overrides

### Requirement: Tail SID diagnosis SHALL emit a Markdown report
The diagnosis SHALL write a human-readable `report.md` that summarizes key diagnosis outputs without replacing the machine-readable CSV and JSON files.

#### Scenario: Markdown report is written
- **WHEN** the diagnosis completes successfully
- **THEN** the output directory MUST contain `report.md`
- **AND** the report MUST include sections for summary, group metrics, top risky items, top risky prefixes, and output files

#### Scenario: Report top-k is configurable
- **WHEN** the diagnosis runner receives `top_k_report`
- **THEN** the generated report MUST limit top risky item and prefix sections to that many rows

### Requirement: Tail SID diagnosis SHALL show report location in stdout
The diagnosis analysis runner SHALL show the Markdown report location and a concise top-risk preview in stdout.

#### Scenario: Report path is displayed
- **WHEN** the diagnosis completes successfully
- **THEN** stdout MUST include the path to `report.md`

#### Scenario: Top risk preview is displayed
- **WHEN** the diagnosis result contains item and prefix risk rows
- **THEN** stdout MUST include the highest-risk item id
- **AND** stdout MUST include the highest-risk prefix

### Requirement: Tail SID diagnosis SHALL use stable damage normalization
Tail-SID diagnosis SHALL compute composite damage scores with a stable normalization strategy that does not amplify near-constant metric components into extremely large scores.

#### Scenario: Degenerate metric component is neutralized
- **WHEN** a metric component has an interquartile range below the configured stability threshold
- **THEN** that component MUST contribute zero to composite `damage`
- **AND** the system MUST NOT divide by a tiny epsilon in a way that produces unbounded scores

#### Scenario: Metric contribution is bounded
- **WHEN** a metric component has non-degenerate spread
- **THEN** its normalized contribution MUST be clamped to a finite configured range before being added to `damage`

#### Scenario: Score metadata is emitted
- **WHEN** diagnosis outputs are written
- **THEN** `summary.json` MUST include score normalization metadata
- **AND** `report.md` MUST describe the normalization method used for `damage`

### Requirement: Tail SID diagnosis SHALL print readable terminal tables
Tail-SID diagnosis SHALL use PrettyTable for human-readable stdout tables while keeping existing machine-readable output files.

#### Scenario: Group metrics are printed with PrettyTable
- **WHEN** diagnosis completes successfully
- **THEN** stdout MUST include a PrettyTable-rendered group metrics table
- **AND** the table MUST include group, item count, full collision rate, strict near-collision rate, local density, and damage columns

#### Scenario: Top risk preview is printed with PrettyTable
- **WHEN** diagnosis result contains item and prefix risk rows
- **THEN** stdout MUST include PrettyTable-rendered top risky item and top risky prefix tables

#### Scenario: Machine-readable outputs are unchanged
- **WHEN** PrettyTable stdout rendering is enabled
- **THEN** `summary.json`, `group_metrics.csv`, `item_damage_scores.csv`, `prefix_risk_scores.csv`, and `report.md` MUST still be written

### Requirement: Tail SID diagnosis SHALL expose metric computation through torchmetrics
Tail-SID diagnosis metric computation SHALL be exposed through a `TailSIDDiagnosisMetric` class that inherits from `torchmetrics.Metric`. The metric SHALL compute the same `DiagnosisResult` structure used by Tail-SID diagnosis reporting without requiring a Lightning trainer or model.

#### Scenario: Diagnosis result is computed through metric lifecycle
- **WHEN** a caller updates `TailSIDDiagnosisMetric` with SID views, training frequencies, frequency groups, and optional embeddings
- **THEN** `compute()` MUST return a `DiagnosisResult`
- **AND** the result MUST include summary, group rows, item rows, and prefix rows

#### Scenario: Metric remains offline-analysis compatible
- **WHEN** the Tail-SID analysis runner computes diagnosis metrics
- **THEN** it MUST instantiate and use `TailSIDDiagnosisMetric`
- **AND** it MUST NOT require `LightningModule`, `Trainer.predict`, or datamodule lifecycle setup
- **AND** it MUST NOT delegate metric orchestration through an intermediate `run_diagnosis(...)` helper or diagnosis config wrapper

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
Tail-SID diagnosis SHALL be launched through the unified Hydra main entrypoint as a `run_mode: analysis` experiment. It SHALL NOT keep an independent argparse CLI as a parallel official entrypoint.

#### Scenario: Diagnosis experiment declares analysis mode
- **WHEN** a maintainer opens `configs/experiment/tail_sid_diagnosis.yaml`
- **THEN** the config MUST declare `run_mode: analysis`
- **AND** it MUST compose its runner from `configs/analysis/tail_sid_diagnosis.yaml`

#### Scenario: Diagnosis script uses unified entrypoint
- **WHEN** a maintainer opens the root diagnosis shell script
- **THEN** it MUST call `uv run --module src.main experiment=tail_sid_diagnosis`
- **AND** it MUST NOT call `src.quantization.tail_sid_diagnosis.run` directly

#### Scenario: Independent CLI is removed
- **WHEN** a maintainer inspects `src/quantization/tail_sid_diagnosis/`
- **THEN** the package MUST NOT expose an argparse-based standalone CLI file for official diagnosis runs

### Requirement: Tail SID diagnosis SHALL use an analysis runner wrapper
Tail-SID diagnosis SHALL expose a Hydra-instantiable analysis runner that wraps the existing diagnosis computation and reporting APIs.

#### Scenario: Diagnosis runner uses existing computation API
- **WHEN** the Tail-SID diagnosis analysis runner executes
- **THEN** it MUST call the existing diagnosis computation path
- **AND** it MUST write the same machine-readable and Markdown outputs as the current diagnosis implementation

#### Scenario: Diagnosis runner receives paths from Hydra config
- **WHEN** the diagnosis analysis runner is instantiated
- **THEN** it MUST receive `data_dir`, `semantic_id_path`, `raw_num_hierarchies`, `output_dir`, and optional `embedding_path` from Hydra config
- **AND** those values MUST be sourced from top-level experiment manual inputs or `paths.output_dir`
