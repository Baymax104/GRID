## Context

`resolve_reference(...)` delegates W&B URI loading to `resolve_wandb_artifact(...)`. Under `torchrun`, every rank executes the same Python entrypoint and instantiates the configured data/model components, so each rank can resolve the same `embedding_path`, `semantic_id_path`, or `ckpt_path` W&B URI.

The current helper computes a deterministic download root but still calls `artifact.download(root=...)` in every process.

## Goals / Non-Goals

**Goals:**
- Avoid repeated W&B artifact downloads in single-node distributed runs.
- Keep W&B URI resolution behavior unchanged for callers.
- Keep the change localized to artifact utility code.

**Non-Goals:**
- Do not add launcher-specific W&B handling.
- Do not solve non-shared filesystem cross-node cache behavior automatically.
- Do not change W&B URI syntax or artifact selection rules.

## Decisions

1. Rank-zero download lives in `src.utils.wandb`

   The duplicated work happens inside artifact resolution, so the fix belongs beside `resolve_wandb_artifact(...)`. This keeps data/model/checkpoint references consistent.

2. Rank 0 downloads, all ranks synchronize, then resolve local file

   When distributed is initialized, rank 0 calls `artifact.download(root=download_root)`. All ranks then pass a distributed barrier and resolve `target_file` under the deterministic `download_root`.

3. Non-distributed behavior remains unchanged

   If torch distributed is not initialized, the helper calls `artifact.download(...)` directly and uses W&B's returned local directory.

4. Shared cache is an explicit assumption

   Non-rank-0 processes read from the same `download_root`. This works for single-node torchrun and any cross-node run that uses a shared cache path. It is not guaranteed for per-node local disks.

## Risks / Trade-offs

- [Risk] Rank 0 download failure can leave other ranks waiting at a barrier -> torchrun should fail the job; robust error broadcast can be added later if needed.
- [Risk] Cross-node non-shared cache will fail on non-rank-0 file lookup -> require shared `GRID_WANDB_ARTIFACT_CACHE` / `wandb_cache_dir` for cross-node runs.
