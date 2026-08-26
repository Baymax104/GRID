## MODIFIED Requirements

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
