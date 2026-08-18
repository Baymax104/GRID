## 1. Remove Runner Lifecycle

- [x] 1.1 Delete `src/common/analysis` offline runner code and obsolete tests.
- [x] 1.2 Delete `configs/analysis/tail_sid_diagnosis.yaml`.
- [x] 1.3 Delete Tail-SID `runner.py` after Lightning components cover its behavior.

## 2. Remove Transitional Logger Support

- [x] 2.1 Remove non-Lightning logger instantiation/finalization helpers that are no longer used.
- [x] 2.2 Keep train/inference logger behavior unchanged after cleanup.

## 3. Residual Reference Cleanup

- [x] 3.1 Remove imports and config references to `run_analysis_runner`, `cfg.analysis.runner`, and Tail-SID runner.
- [x] 3.2 Run residual reference scans for offline analysis runner paths.
- [x] 3.3 Run focused pytest, scoped Ruff, and OpenSpec validation.
