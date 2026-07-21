## 1. Config cleanup

- [x] 1.1 Remove `user_id` from `configs/data/tiger_train.yaml` training preprocessing inputs when it is not needed by labels or model inputs.
- [x] 1.2 Remove `user_id: user_id` from TIGER train and inference `feature_to_model_input_map`.
- [x] 1.3 Remove `num_user_bins` from TIGER train and inference model configs.
- [x] 1.4 Keep `user_id` in `configs/data/tiger_inference.yaml` only where needed for output identity keys.

## 2. Inference collate behavior

- [x] 2.1 Update `collate_fn_inference_for_sequence` so `id_field_name` fields populate `user_id_list` only.
- [x] 2.2 Ensure id fields are not padded/trimmed into `transformed_sequences`.
- [x] 2.3 Ensure `mask` is still computed from the first non-id sequence field.

## 3. TIGER model cleanup

- [x] 3.1 Remove `num_user_bins` from `SemanticIDEncoderDecoder.__init__`.
- [x] 3.2 Remove `self.user_embedding` creation and storage.
- [x] 3.3 Remove `user_id` parameters from `encoder_forward_pass`, `generate`, and `forward`.
- [x] 3.4 Remove the user embedding prepend branch and stale comments that assume an extra user token.
- [x] 3.5 Ensure inference `model_step` no longer passes `user_id` to generation.

## 4. Verification

- [x] 4.1 Search configs and TIGER code to confirm no `num_user_bins` or `feature_to_model_input_map.user_id` remains.
- [x] 4.2 Run `uv run python -m compileall -q src/data/components/collate.py src/recommendation/tiger_generation_model.py`.
- [x] 4.3 Run `uv run ruff check src/data/components/collate.py src/recommendation/tiger_generation_model.py`.
- [x] 4.4 Hydra compose and instantiate `experiment=tiger_train` and `experiment=tiger_inference` with a small semantic ID bundle.
- [x] 4.5 Smoke test `collate_fn_inference_for_sequence`: `user_id_list` is populated, `transformed_sequences` excludes `user_id`, and mask shape follows `sequence_data`.
- [x] 4.6 Smoke test TIGER `predict_step`: `ModelOutput.keys` matches `batch.user_id_list` and predictions shape is unchanged.
- [x] 4.7 Run `openspec validate remove-user-id-training-path --strict`.
