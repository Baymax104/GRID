## Context

Tail-SID diagnosis currently runs as an official `run_mode: analysis` experiment through `cfg.analysis.runner`. That path was intentionally offline, but the analysis experiment now needs the same lifecycle features already provided by `pipeline_launcher`: DataModule setup, LightningModule execution, Trainer-managed test lifecycle, logger configuration, callbacks, and W&B artifact handling.

The target semantic is test-only analysis. Analysis jobs do not train parameters and do not produce inference predictions; they evaluate existing artifacts/data and emit metrics plus diagnosis outputs.

## Goals / Non-Goals

**Goals:**
- Make official analysis experiments use the existing Lightning component assembly path.
- Execute analysis through `Trainer.test(...)`.
- Add Tail-SID DataModule, LightningModule, and callback components.
- Preserve Tail-SID summary, CSV, and Markdown output schemas.
- Keep W&B usage at the logger/callback layer instead of domain metric code.

**Non-Goals:**
- Do not add a training step or optimizer for Tail-SID diagnosis.
- Do not change Tail-SID metric formulas or report schemas.
- Do not change inference experiments to use W&B.
- Do not introduce a new external logging dependency.

## Decisions

1. Analysis uses `Trainer.test(...)`

   `test` is the closest Lightning lifecycle for diagnosis: it consumes data, computes evaluation-style metrics, emits logger metrics, and runs callbacks without training. `predict` is less appropriate because Tail-SID diagnosis produces reports and metrics, not model predictions.

2. Tail-SID DataModule returns one full analysis batch first

   The current diagnosis algorithm needs global SID prefix buckets, global frequency groups, and optional global embedding alignment. A single-batch DataModule preserves existing behavior while placing data loading under the Lightning lifecycle. If future analysis becomes batch-reducible, the DataModule can evolve without changing the main entrypoint contract.

3. Tail-SID LightningModule owns metric computation, callback owns reporting/artifacts

   The module should load no W&B APIs and should focus on computing a `DiagnosisResult` during test. The callback should write local outputs via existing reporting functions and, when the active logger supports it, attach those files to the run.

4. Numeric summary metrics are logged from the module

   The module will filter `DiagnosisResult.summary` to numeric values and call `self.log_dict(...)` so W&B and any future Lightning logger receive scalar metrics through the standard path.

## Risks / Trade-offs

- Single-batch loading can hold large SID/embedding artifacts in memory -> this matches the current diagnosis behavior and avoids changing metric semantics during the migration.
- Lightning test introduces Trainer overhead for simple offline analysis -> accepted because the project now values unified data/logger/callback semantics over a separate lightweight runner.
- W&B artifact APIs are logger-specific -> isolate optional artifact upload in the callback and keep local files as the source of truth.

## Migration Plan

1. Add Tail-SID DataModule, LightningModule, and report callback.
2. Switch `run_mode: analysis` to call `Trainer.test(...)`.
3. Convert `tail_sid_diagnosis` config to compose `data`, `model`, `callbacks`, `logger`, and `trainer`.
4. Add tests for Hydra composition, test-only execution, metrics logging, and report file generation.
5. After the Lightning path is verified, run the cleanup change to remove the old offline runner path.
