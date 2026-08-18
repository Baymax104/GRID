## MODIFIED Requirements

### Requirement: Shared datamodule mechanics SHALL be centralized in the base class
The shared datamodule implementation SHALL centralize stage lifecycle mechanics separately from concrete loading mechanics. File-backed loading SHALL use `FileDataModule`, while artifact-backed diagnosis loading SHALL use `DiagnosisDataModule`.

#### Scenario: Diagnosis datamodule is inspected
- **WHEN** developers inspect the data-layer datamodule implementations
- **THEN** `DiagnosisDataModule` MUST live under `src.data.datamodule`
- **AND** it MUST inherit the shared stage lifecycle base
- **AND** it MUST NOT depend on Tail-SID metric modules

#### Scenario: Diagnosis datamodule builds a test dataloader
- **WHEN** `DiagnosisDataModule` receives `setup("test")`
- **THEN** it MUST instantiate its configured diagnosis dataset for the testing stage
- **AND** `test_dataloader()` MUST return a dataloader over that dataset
