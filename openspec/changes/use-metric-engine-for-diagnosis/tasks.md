## Tasks

- [x] Change `TailSIDDiagnosisModule.test_step` to return expanded metric pre-state fields.
- [x] Refactor diagnosis metrics so structural, semantic, damage, and prefix metrics update independently from pre-state fields.
- [x] Add config-driven `MetricEngine` metrics for Tail-SID diagnosis test stage without adapters.
- [x] Add a diagnosis result callback that assembles `DiagnosisResult`, stores it on the module, and logs numeric summary values.
- [x] Keep the report callback focused on output files and logger artifact save.
- [x] Update focused unit tests and Hydra config assertions.
- [x] Run focused pytest, Ruff, residual scans, and OpenSpec validation.
