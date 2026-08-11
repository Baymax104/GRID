## 1. Common Analysis Runtime

- [x] 1.1 Add `src/common/analysis/` with a lightweight runner protocol and Hydra instantiation helper.
- [x] 1.2 Add `run_analysis(cfg)` to `src/main.py` and dispatch `run_mode: analysis`.
- [x] 1.3 Adjust config warning logic so analysis experiments do not require model/trainer-only config groups.

## 2. Hydra Config Structure

- [x] 2.1 Add `configs/analysis/` as the analysis runner component group.
- [x] 2.2 Add `configs/experiment/tail_sid_diagnosis.yaml` with `run_mode: analysis` and top-level manual inputs.
- [x] 2.3 Add `configs/analysis/tail_sid_diagnosis.yaml` that instantiates the diagnosis runner from top-level inputs.

## 3. Tail SID Diagnosis Migration

- [x] 3.1 Add a Hydra-instantiable Tail-SID diagnosis runner wrapper under `src/quantization/tail_sid_diagnosis/`.
- [x] 3.2 Remove the independent argparse CLI entrypoint for official diagnosis runs.
- [x] 3.3 Update root `tail_sid_diagnosis.sh` to call `uv run --module src.main experiment=tail_sid_diagnosis`.

## 4. Data/Common Boundary

- [x] 4.1 Keep sequence frequency scanning on reusable data-domain readers/helpers rather than Lightning DataModule.
- [x] 4.2 Ensure analysis-specific orchestration remains outside `src/data`.

## 5. Verification

- [x] 5.1 Add unit tests for analysis runner instantiation and `run_mode: analysis` dispatch.
- [x] 5.2 Add Hydra composition smoke test for `experiment=tail_sid_diagnosis`.
- [x] 5.3 Update diagnosis tests for runner-based execution and removal of CLI help assertions.
- [x] 5.4 Run focused pytest, ruff, and `openspec validate add-analysis-run-mode --strict`.
