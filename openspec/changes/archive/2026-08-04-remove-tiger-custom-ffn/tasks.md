## 1. Model Configuration

- [x] 1.1 Remove `mlp_layers` from `configs/model/tiger_train.yaml`.
- [x] 1.2 Remove `mlp_layers` from `configs/model/tiger_inference.yaml`.

## 2. Model Implementation

- [x] 2.1 Remove the `mlp_layers` constructor parameter and T5 FFN replacement block from `SemanticIDEncoderDecoder`.
- [x] 2.2 Delete the now-unused `src/recommendation/t5_multi_layer_ff.py` module.
- [x] 2.3 Confirm no active references to `T5MultiLayerFF`, `t5_multi_layer_ff`, or `mlp_layers` remain outside the change artifacts.

## 3. Specification and Verification

- [x] 3.1 Update the living `self-contained-tiger-generation-model` spec with the custom FFN removal requirement.
- [x] 3.2 Validate `remove-tiger-custom-ffn` with OpenSpec strict validation.
- [x] 3.3 Run scoped Python lint checks for touched TIGER code.
- [x] 3.4 Run a TIGER config compose or instantiation smoke check that verifies train/inference configs no longer require `mlp_layers`.
