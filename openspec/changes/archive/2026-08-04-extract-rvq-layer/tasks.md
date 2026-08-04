## 1. Layer Module

- [x] 1.1 Add `VectorQuantizationLayer` under `src/quantization/rvq/` with one codebook parameter, initialization buffer, initialized flag, reset, training forward, and predict methods.
- [x] 1.2 Move RVQ single-layer K-Means++ initialization and VQ-STE quantization behavior into the layer module without introducing cross-model shared abstractions.

## 2. ResidualVectorQuantization Refactor

- [x] 2.1 Change `ResidualVectorQuantization` to accept `sub_layer` and construct `self.layers` as one layer module per hierarchy.
- [x] 2.2 Replace model-level `centroids_list`, `init_buffers`, and `is_initialized_list` usages with layer-owned state.
- [x] 2.3 Simplify layer-wise training state to `current_layer` plus `layer_step_boundaries` and update train-start and layer-advance logic.
- [x] 2.4 Update output statistics and checkpoint save/load metadata for module-owned centroids and initialized flags.

## 3. Configuration and Tests

- [x] 3.1 Update `configs/model/rvq_train.yaml` to declare the RVQ layer through `sub_layer`.
- [x] 3.2 Update existing quantization tests and add focused RVQ layer/module assertions for new parameter ownership and behavior.

## 4. Verification

- [x] 4.1 Run the focused quantization test file with `uv run pytest tests/quantization/rkmeans/test_kmeans_layer.py`.
- [x] 4.2 Run OpenSpec validation/status checks for `extract-rvq-layer` if available.
