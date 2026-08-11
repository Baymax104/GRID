## Context

Tail-SID diagnosis is an offline analysis runner. It loads keyed Semantic ID and optional embedding bundles, scans training records for frequency groups, computes diagnosis rows, and writes report artifacts. The metric calculation currently lives mostly in module-level functions, with `compute_metrics(...)` serving as the public entrypoint.

The desired change is structural: make the metric computation owned by a torchmetrics-compatible class while keeping the analysis runner independent from the Lightning trainer lifecycle.

## Goals / Non-Goals

**Goals:**

- Introduce `TailSIDDiagnosisMetric` as the single public metric entrypoint for computing `DiagnosisResult`.
- Move diagnosis metric state and major calculation steps into class methods.
- Keep Tail-SID analysis orchestration in `TailSIDDiagnosisRunner.run()` instead of delegating to an extra diagnosis function/config layer.
- Keep `DiagnosisResult` output shape and existing output file behavior unchanged.
- Remove the public `compute_metrics(...)` function so new callers use the metric object directly.
- Remove the public `run_diagnosis(...)` helper and `DiagnosisConfig` wrapper.

**Non-Goals:**

- Do not convert the analysis experiment into a `LightningModule` or `Trainer.predict` pipeline.
- Do not split every item-level field into independent torchmetrics metrics.
- Do not change scoring formulas, output filenames, or report schemas.
- Do not introduce new dependencies.

## Decisions

### 1. Use `torchmetrics.Metric` for the metric boundary

`TailSIDDiagnosisMetric` will inherit `torchmetrics.Metric` and expose `update(...)` plus `compute()`. This gives the diagnosis metric a familiar lifecycle and future compatibility with the project's metric direction.

Alternative considered: use a plain service class such as `TailSIDDiagnosisAnalyzer`. That would structure the code, but it would not align with the user's goal of introducing a torchmetrics-style metric boundary.

### 2. Store one complete diagnosis input per metric instance

The metric will accept a complete `SIDViews`, frequency map, group map, and optional embedding tensor in `update(...)`, then compute one `DiagnosisResult`. This matches current offline semantics and avoids partial aggregation rules for dict/dataclass states.

Alternative considered: incremental row-by-row updates. That would require reworking prefix bucket construction and semantic mismatch computation, and it is not needed for the current offline analysis.

### 3. Keep helpers private to the metric class

Current metric helpers such as prefix bucket construction, semantic mismatch, group aggregation, summary construction, and robust score normalization will become private methods on `TailSIDDiagnosisMetric`. This makes the class the clear metric owner without creating a single oversized `compute()` body.

Alternative considered: keep all helpers as module-level functions and have the metric call them. That preserves pure functions but weakens the requested structural boundary.

### 4. Keep IO and reporting outside the metric

Input loading remains in `io.py`, and output writing remains in `reporting.py`. The metric computes `DiagnosisResult` only; it does not read files or write reports.

Alternative considered: move the entire diagnosis flow into the metric. That would couple metric computation to filesystem side effects and make testing harder.

### 5. Let the runner own orchestration directly

`TailSIDDiagnosisRunner.run()` will load SID views, load optional embeddings, compute training frequencies, assign frequency groups, update/compute `TailSIDDiagnosisMetric`, and write reports directly from its Hydra-provided fields. This removes the redundant `run_diagnosis(...)` helper and `DiagnosisConfig` wrapper.

Alternative considered: keep `run_diagnosis(config)` as a service boundary. That added indirection without owning a distinct responsibility, because the runner already carries the same validated Hydra fields.

## Risks / Trade-offs

- [Risk] `torchmetrics.Metric` usually targets tensor states and scalar outputs, while this diagnosis returns a report dataclass. → Mitigation: keep usage offline and explicitly test direct `update()` / `compute()` behavior instead of relying on Lightning logger integration.
- [Risk] Removing `compute_metrics(...)`, `run_diagnosis(...)`, and `DiagnosisConfig` is a breaking internal API change for tests and any ad hoc imports. → Mitigation: update package exports and tests in the same change.
- [Risk] Moving helper functions into a class may reduce direct unit-test granularity. → Mitigation: test through the public metric entrypoint and preserve coverage for representative collision, normalization, and mismatch behavior.
