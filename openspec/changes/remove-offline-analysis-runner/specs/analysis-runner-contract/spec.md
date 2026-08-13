## REMOVED Requirements

### Requirement: Analysis runners SHALL execute through a common runner contract
**Reason**: Official analysis experiments now execute through the Lightning test pipeline, which provides DataModule, LightningModule, Trainer, callback, and logger lifecycle support.

**Migration**: Use `run_mode: analysis` with `pipeline_launcher` and `Trainer.test(...)`. Analysis-specific computation belongs in a LightningModule and analysis outputs belong in callbacks or reporting helpers.

### Requirement: Analysis components SHALL live in common analysis and domain packages
**Reason**: The common offline analysis lifecycle is removed. Domain-specific analysis code remains in its owning package, while shared lifecycle behavior is provided by the existing Lightning launcher.

**Migration**: Place Tail-SID analysis DataModule, LightningModule, and callbacks under `src/quantization/tail_sid_diagnosis/`.

### Requirement: Analysis runners MAY reuse data-domain readers and datasets
**Reason**: Official analysis experiments now reuse data loading through Lightning DataModules instead of custom runner-owned orchestration.

**Migration**: Implement analysis DataModules that can reuse existing data-domain helpers internally.
