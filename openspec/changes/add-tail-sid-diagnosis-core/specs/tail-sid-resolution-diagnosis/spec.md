## ADDED Requirements

### Requirement: Tail SID diagnosis SHALL load keyed bundle inputs
The system SHALL provide an offline Tail-SID Resolution Damage diagnosis entrypoint that loads Semantic ID predictions from a keyed prediction bundle and optionally loads item embeddings from a keyed prediction bundle. The diagnosis SHALL NOT assume item ids are contiguous tensor row indexes.

#### Scenario: Semantic ID bundle is loaded by key
- **WHEN** the diagnosis CLI receives `--semantic-id-path`
- **THEN** it MUST load the file through the keyed prediction bundle loader
- **AND** it MUST preserve the item keys associated with each SID row

#### Scenario: Raw and model SID views are retained
- **WHEN** the SID prediction width is greater than `--raw-num-hierarchies`
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
The diagnosis SHALL create a clear output directory containing summary, group metrics, item damage scores, and prefix risk scores. The CLI SHALL also print a concise human-readable run summary.

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
The project SHALL provide a root-level script for running the diagnosis from the repository root using `uv run`.

#### Scenario: Root script documents required arguments
- **WHEN** a maintainer opens the root diagnosis script
- **THEN** it MUST call the diagnosis module with `uv run`
- **AND** it MUST expose `data_dir`, `semantic_id_path`, `raw_num_hierarchies`, and `output_dir` as editable arguments

#### Scenario: Diagnosis module supports help output
- **WHEN** a maintainer runs the diagnosis module with `--help`
- **THEN** the command MUST exit successfully
- **AND** it MUST list the required inputs
