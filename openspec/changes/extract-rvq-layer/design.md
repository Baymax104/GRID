## Context

`ResidualVectorQuantization` currently keeps each hierarchy's codebook parameter and runtime initialization state in parallel model-level lists. The single-layer behavior is implemented by model methods that take `layer_idx`, while `ResidualKMeans` already delegates similar single-layer behavior to `KMeansLayer` modules stored in `self.layers`.

This change keeps RVQ independent within `src/quantization/rvq/` while moving one hierarchy's VQ-STE codebook behavior into a dedicated `nn.Module`.

## Goals / Non-Goals

**Goals:**
- Make each RVQ hierarchy an independent layer module owning exactly one codebook parameter and runtime initialization state.
- Keep `ResidualVectorQuantization` responsible for residual traversal, Lightning lifecycle, metrics, optimizer configuration, and layer-wise schedule.
- Simplify RVQ layer scheduling to a linear `current_layer` plus `layer_step_boundaries` model, matching RKMeans where practical.
- Update configuration and tests to reflect module-owned codebooks.

**Non-Goals:**
- Do not preserve compatibility with old RVQ checkpoints using `centroids_list.*` keys.
- Do not refactor RQVAE in this change.
- Do not introduce a cross-model shared quantization base class.

## Decisions

1. **Create `VectorQuantizationLayer` under `src/quantization/rvq/`.**
   - Rationale: keeps RVQ self-contained in its model directory and avoids reintroducing deleted shared abstractions.
   - Alternative considered: reuse or revive a shared `VectorQuantization` abstraction. Rejected because current specs require independent quantization models and removal of old shared abstractions.

2. **Inject RVQ layers through `sub_layer`, matching RKMeans.**
   - Rationale: configuration declares the layer implementation and constructor parameters in the same style as `rkmeans_train.yaml`.
   - Alternative considered: instantiate `VectorQuantizationLayer` directly inside RVQ. Rejected because it makes RVQ less consistent with RKMeans and less configurable for tests.

3. **Move only single-layer state and quantization behavior into the layer module.**
   - The layer owns `centroids`, `init_buffer`, `is_initialized`, initialization, VQ-STE training forward inputs, and prediction.
   - The parent model owns residual normalization, residual subtraction, quantization loss calculation, output stacking, metrics, checkpoint metadata, and step schedule.

4. **Use new checkpoint structure only.**
   - Rationale: the user confirmed old checkpoint compatibility is not required.
   - New codebook keys will be `layers.<index>.centroids`.

## Risks / Trade-offs

- **Checkpoint breakage for old RVQ runs** → Accepted; old checkpoint migration is explicitly out of scope.
- **Hydra config mismatch if `sub_layer` is missing** → Mitigated by updating `configs/model/rvq_train.yaml` and tests to instantiate through a helper.
- **Layer initialization state not persisted as parameters** → Mitigated by keeping `layers_initialized` checkpoint metadata and restoring each layer's flag in `on_load_checkpoint`.
