## 1. Spec And Structure

- [x] 1.1 Validate the OpenSpec proposal, design, and Tail-SID spec delta.
- [x] 1.2 Introduce shared diagnosis context and metric output dataclasses.

## 2. Metric Split

- [x] 2.1 Split structural item metric computation into an independent metric class.
- [x] 2.2 Split semantic mismatch computation into an independent metric class.
- [x] 2.3 Split damage score computation into an independent metric class.
- [x] 2.4 Split prefix risk computation into an independent metric class and keep group/summary assembly pure.

## 3. Module Orchestration

- [x] 3.1 Update `TailSIDDiagnosisModule.test_step` to build shared context once.
- [x] 3.2 Update `TailSIDDiagnosisModule.test_step` to compose split metric outputs into `DiagnosisResult`.

## 4. Verification

- [x] 4.1 Update Tail-SID tests for split metrics and module orchestration.
- [x] 4.2 Run focused Tail-SID pytest.
- [x] 4.3 Run scoped Ruff and OpenSpec validation.
