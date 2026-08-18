## REMOVED Requirements

### Requirement: Tail SID diagnosis SHALL use an analysis runner wrapper
**Reason**: Tail-SID diagnosis now runs through a Lightning DataModule, LightningModule, callback, Trainer, and logger lifecycle.

**Migration**: Use the Tail-SID diagnosis analysis experiment with `run_mode: analysis`; the module computes the diagnosis during `Trainer.test(...)` and the callback writes the existing report outputs.

## MODIFIED Requirements

### Requirement: Tail SID diagnosis SHALL run as an official analysis experiment
Tail-SID diagnosis SHALL be launched through the unified Hydra main entrypoint as a `run_mode: analysis` experiment. It SHALL run through the Lightning test lifecycle and SHALL NOT keep an independent argparse CLI or Hydra analysis-runner wrapper as a parallel official entrypoint.

#### Scenario: Diagnosis experiment declares analysis mode
- **WHEN** a maintainer opens `configs/experiment/tail_sid_diagnosis.yaml`
- **THEN** the config MUST declare `run_mode: analysis`
- **AND** it MUST compose Lightning component configs rather than `/analysis@analysis`

#### Scenario: Diagnosis script uses unified entrypoint
- **WHEN** a maintainer opens the root diagnosis shell script
- **THEN** it MUST call `uv run --module src.main experiment=tail_sid_diagnosis`
- **AND** it MUST NOT call `src.quantization.tail_sid_diagnosis.run` directly

#### Scenario: Independent CLI is removed
- **WHEN** a maintainer inspects `src/quantization/tail_sid_diagnosis/`
- **THEN** the package MUST NOT expose an argparse-based standalone CLI file for official diagnosis runs
