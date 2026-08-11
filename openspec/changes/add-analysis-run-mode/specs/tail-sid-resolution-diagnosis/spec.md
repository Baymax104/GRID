## ADDED Requirements

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
