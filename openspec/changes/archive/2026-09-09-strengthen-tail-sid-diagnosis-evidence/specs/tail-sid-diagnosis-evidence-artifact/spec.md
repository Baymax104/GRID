## ADDED Requirements

### Requirement: Diagnosis evidence SHALL be written as a stable structured output
The system SHALL write each completed official Tail-SID diagnosis to a dedicated local analysis directory using JSON and CSV files with stable schemas. The output SHALL contain run-level summary and verdict data plus group-, item-, prefix-, and bounded harmful-pair evidence needed for audit and downstream methods.

#### Scenario: Complete structural diagnosis writes required files
- **WHEN** Tail-SID diagnosis completes without recommendation outcomes
- **THEN** the output directory MUST contain `summary.json`, `group_metrics.csv`, `item_damage_scores.csv`, `prefix_risk_scores.csv`, and `harmful_overlap_pairs.csv`
- **AND** `summary.json` MUST identify recommendation evidence as unavailable

#### Scenario: Recommendation evidence extends item outputs
- **WHEN** a recommendation output bundle is provided
- **THEN** item evidence MUST include label support and configured hit/rank outcome fields
- **AND** summary output MUST include the configured raw-damage correlation and frequency-matched comparison results

### Requirement: Diagnosis evidence schemas SHALL preserve raw and prioritized meanings
Evidence files SHALL use distinct fields for observed raw risk and frequency-aware priority. Item and prefix schemas MUST retain keyed IDs and the raw/model SID distinction so consumers do not infer row-index identity or mistake de-duplication for raw quantizer structure.

#### Scenario: Item evidence is keyed and auditable
- **WHEN** an item evidence row is written
- **THEN** it MUST include `item_id`, frequency group, training frequency, raw SID, model SID, de-duplication digit, raw component values, `raw_damage`, and `priority_score`
- **AND** the row order MUST NOT define item identity

#### Scenario: Prefix evidence supports downstream repair
- **WHEN** a prefix evidence row is written
- **THEN** it MUST include prefix depth/value, bucket size and group composition, suffix uniqueness, raw risk aggregates, semantic evidence availability, and separately named raw and priority prefix scores

### Requirement: Harmful-pair output SHALL be deterministic and bounded
The diagnosis SHALL bound harmful-pair output by configured global and/or per-item limits while preserving deterministic selection. Metadata MUST record the limits, sampling seed, selection rule, and whether any rows were truncated.

#### Scenario: Large pair sets cannot create unbounded output
- **WHEN** candidate harmful pairs exceed configured limits
- **THEN** the writer MUST retain pairs using the configured deterministic ranking and per-item/global limits
- **AND** summary metadata MUST report candidate, retained, and truncated counts

### Requirement: Diagnosis evidence SHALL be publishable as one W&B Artifact
When a W&B logger run is configured, the official diagnosis SHALL publish the complete local evidence directory as one Artifact through the logger-owned run. Publishing MUST NOT initialize, finish, or replace the logger run, and local-only output MUST remain usable without W&B.

#### Scenario: Logger-owned run publishes complete evidence
- **WHEN** diagnosis completes with an active W&B logger run
- **THEN** one Artifact of the configured diagnosis evidence type MUST contain every required evidence file
- **AND** Artifact metadata MUST record task, dataset, tokenizer input references, schema version, primary analysis settings, and verdict

#### Scenario: Local-only output does not require W&B
- **WHEN** no W&B logger is configured and local output is enabled
- **THEN** the complete evidence directory MUST still be written
- **AND** the writer MUST NOT initialize a W&B run

### Requirement: Diagnosis evidence output SHALL be atomic and fail visibly
The writer SHALL stage files before exposing a completed local output and SHALL propagate serialization or Artifact publication failures. A failed output MUST NOT be presented as a complete evidence Artifact.

#### Scenario: Serialization failure does not publish partial evidence
- **WHEN** a required evidence file cannot be serialized
- **THEN** the diagnosis run MUST fail before Artifact publication
- **AND** no completion marker or successful verdict file MUST be exposed

#### Scenario: Artifact upload failure is not swallowed
- **WHEN** W&B Artifact publication is configured and upload fails
- **THEN** the failure MUST propagate to the official diagnosis run
- **AND** the logger-owned run lifecycle MUST remain owned by the configured logger

### Requirement: Diagnosis evidence SHALL preserve module ownership and dependency direction
The evidence implementation SHALL keep input resolution and keyed assembly in the data layer, Tail-SID formulas and verdicts in the diagnosis domain, scalar lifecycle in the shared metric runtime, structured serialization/publication in common writers, and input lineage in the existing lineage callback. Shared infrastructure MUST consume domain-neutral protocols and MUST NOT import the Tail-SID diagnosis package.

#### Scenario: Data assembly does not compute diagnosis evidence
- **WHEN** semantic ID, embedding, testing label, or recommendation inputs are loaded and aligned
- **THEN** data-layer components MUST return keyed raw/runtime data without computing damage, statistical evidence, or verdicts
- **AND** W&B input resolution MUST continue through the shared Artifact resolver

#### Scenario: Common writer is domain neutral
- **WHEN** diagnosis evidence is handed to a common structured writer
- **THEN** the diagnosis domain MUST first adapt it to a generic named-document/named-table payload
- **AND** the common writer MUST NOT import diagnosis evidence types or branch on Tail-SID field names

#### Scenario: Existing lifecycle owners remain unchanged
- **WHEN** the official diagnosis is assembled and executed
- **THEN** the launcher and metric callback MUST remain free of diagnosis-specific conditional paths
- **AND** input `use_artifact` calls MUST remain owned by `WandbArtifactLineageCallback`
- **AND** run initialization/finalization MUST remain owned by configured loggers

#### Scenario: Unified entrypoint and Hydra output ownership are preserved
- **WHEN** diagnosis evidence is produced
- **THEN** execution MUST continue through the root script, `src.main`, unified launcher, and `Trainer.test`
- **AND** local output paths MUST derive from configured `paths.output_dir`
