## ADDED Requirements

### Requirement: Downstream W&B scripts SHALL require a quantization method group

The official `tiger_train.sh`, `tiger_inference.sh`, and `tail_sid_diagnosis.sh` scripts SHALL require callers to select the quantization-method W&B group and SHALL forward that value to the top-level Hydra `group` field.

#### Scenario: Group is passed with equals syntax
- **WHEN** a user invokes a downstream script with `--group=rkmeans`
- **THEN** the script MUST pass `group=rkmeans` to the unified entrypoint

#### Scenario: Group is passed with separated syntax
- **WHEN** a user invokes a downstream script with `--group rvq`
- **THEN** the script MUST pass `group=rvq` as one Hydra argument

#### Scenario: RQ-VAE downstream run is grouped explicitly
- **WHEN** a user invokes a downstream script with `--group rqvae`
- **THEN** the script MUST pass `group=rqvae` to the unified entrypoint

#### Scenario: Missing group fails fast
- **WHEN** a user invokes a downstream script without `--group` or provides the flag without a value
- **THEN** the script MUST exit with status 2
- **AND** the error MUST state that `--group` requires `rkmeans`, `rvq`, or `rqvae`

#### Scenario: Unsupported group fails fast
- **WHEN** a user invokes a downstream script with a group other than `rkmeans`, `rvq`, or `rqvae`
- **THEN** the script MUST exit with status 2
- **AND** it MUST NOT invoke the unified Python entrypoint

#### Scenario: Group preserves existing script contracts
- **WHEN** a downstream script receives group together with dry-run, notes, checkpoint or artifact paths, and extra Hydra overrides supported by that script
- **THEN** every supported option MUST remain intact
- **AND** extra Hydra overrides MUST still appear after the script's default `group` argument so explicit trailing overrides retain precedence
