## Context

The Lightning Tail-SID analysis module currently instantiates one `TailSIDDiagnosisMetric` in `test_step`. That metric builds prefix buckets and group indexes internally, computes every metric family, assembles rows, builds summary fields, and returns a full `DiagnosisResult`.

After moving analysis into Lightning, `test_step` is the right orchestration boundary: it receives the complete analysis batch, can build shared state once, and can compose independent metric components.

## Goals / Non-Goals

**Goals:**
- Build shared diagnosis context once in `TailSIDDiagnosisModule.test_step`.
- Split independent metric families into focused `torchmetrics.Metric` classes.
- Preserve existing `DiagnosisResult`, output file schemas, and scalar logging behavior.
- Keep the code easy to test at metric-family granularity.

**Non-Goals:**
- Do not change Tail-SID formulas.
- Do not change DataModule loading, callback reporting, W&B config, or Hydra component wiring.
- Do not add batch-reduction/distributed aggregation semantics in this change.

## Decisions

1. Introduce `DiagnosisContext`

   `DiagnosisContext` contains shared derived state: item ids, raw/model SID views, group indexes, prefix buckets, strict depth, and SID lengths. `test_step` builds it once and passes it to metric components.

2. Keep metric components local to `metrics.py`

   The split is about responsibility boundaries, not package sprawl. Keeping related metric classes in `metrics.py` avoids scattering formula code while still making the components independently testable.

3. Keep row/summary assembly as pure functions

   Row builders are schema assembly, not torchmetrics state. They should remain pure functions that consume metric outputs and context.

4. Remove the old aggregate `TailSIDDiagnosisMetric`

   The Lightning module and tests should use the shared context plus split metrics directly. Keeping a public facade would preserve the old orchestration surface and make it unclear where shared state is meant to be built.

## Risks / Trade-offs

- More classes can increase surface area -> keep constructor parameters minimal and outputs dataclass-based.
- Existing tests import `TailSIDDiagnosisMetric` -> update tests to cover new components and remove the old aggregate metric import.
- The living spec is stale until prior changes are archived -> this change writes a delta that matches the current working-tree direction and avoids reviving the old runner path.
