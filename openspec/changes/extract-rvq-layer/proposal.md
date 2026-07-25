## Why

ResidualVectorQuantization currently owns per-layer codebook parameters, initialization buffers, initialization state, and single-layer VQ-STE behavior directly in the LightningModule. RKMeans has already moved equivalent single-layer responsibilities into a dedicated `nn.Module`, and RVQ should follow the same structure to reduce model-level state and make layer behavior independently testable.

## What Changes

- Add a dedicated RVQ single-layer `nn.Module` that owns one codebook, initialization buffer, initialized flag, VQ-STE training operation, and prediction operation.
- Change `ResidualVectorQuantization` to own an `nn.ModuleList` of RVQ layer modules instead of model-level `centroids_list`, `init_buffers`, and `is_initialized_list`.
- Simplify RVQ layer-wise training control to the same linear current-layer and step-boundary shape used by RKMeans.
- Update RVQ model configuration to inject the layer module through `sub_layer`, matching the RKMeans configuration pattern.
- **BREAKING**: New RVQ checkpoints store codebook parameters under `layers.<index>.centroids`; compatibility with old `centroids_list.*` checkpoint keys is not required.

## Capabilities

### New Capabilities
- `rvq-vector-quantization-layer-modules`: Defines the module ownership and training protocol for RVQ single-layer vector quantization modules.

### Modified Capabilities
- `independent-quantization-models`: RVQ may delegate single-layer quantization behavior to a dedicated module in its model directory while remaining independent and self-contained within `src/quantization/rvq/`.

## Impact

- Affected code: `src/quantization/rvq/residual_vector_quantization.py`, new RVQ layer module, RVQ model config, quantization tests.
- Affected checkpoint structure: RVQ centroid parameter names change from `centroids_list.<index>` to `layers.<index>.centroids`.
- No new dependencies.
