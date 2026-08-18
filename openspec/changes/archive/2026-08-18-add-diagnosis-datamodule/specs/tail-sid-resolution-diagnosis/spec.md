## MODIFIED Requirements

### Requirement: Tail SID diagnosis SHALL derive frequency groups from training data
The diagnosis SHALL scan `data_dir/training` sequence records to compute training frequency per item and assign each known item to Head, Mid, Tail, or Tail-Cold groups without using evaluation or testing labels for the primary split. This frequency grouping SHALL be expressed as data preprocessing for the diagnosis dataset.

#### Scenario: Training sequence frequencies are grouped
- **WHEN** the diagnosis dataset loads semantic IDs and training sequence records
- **THEN** it MUST compute `freq_train` from `sequence_data`
- **AND** a configured preprocessing function MUST assign non-cold items by configured head and tail ratios
- **AND** it MUST assign known items with zero training frequency to `Tail-Cold`

### Requirement: Tail SID diagnosis SHALL run as an official analysis experiment
Tail-SID diagnosis SHALL be launched through the unified Hydra main entrypoint as a `run_mode: analysis` experiment and SHALL load its test inputs through the shared diagnosis DataModule/Dataset path.

#### Scenario: Diagnosis experiment declares data-layer diagnosis datamodule
- **WHEN** a maintainer composes `experiment=tail_sid_diagnosis`
- **THEN** the data datamodule target MUST be `src.data.datamodule.DiagnosisDataModule`
- **AND** the test dataset target MUST be the shared diagnosis dataset
- **AND** the quantization package MUST NOT define a Tail-SID-specific DataModule for official runs
