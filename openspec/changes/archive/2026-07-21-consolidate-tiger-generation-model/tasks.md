## 1. Semantic ID Artifact Contract

- [x] 1.1 Add a model-side semantic ID tensor loader that extracts `predictions` from the keyed semantic ID bundle and validates tensor rank/shape.
- [x] 1.2 Update `configs/model/tiger_train.yaml` to pass `semantic_ids` from the new tensor loader instead of `codebooks: load_model_output(...)`.
- [x] 1.3 Update `configs/model/tiger_inference.yaml` to pass `semantic_ids` from the new tensor loader instead of `codebooks: load_model_output(...)`.
- [x] 1.4 Confirm data-side preprocessing still receives the full keyed `semantic_id_bundle` for key-based lookup.

## 2. Self-contained TIGER Model

- [x] 2.1 Make `SemanticIDEncoderDecoder` directly inherit from LightningModule and own its train/eval/test/predict runtime behavior.
- [x] 2.2 Move TIGER-specific semantic ID state, prefix validation, deterministic hooks, beam search, and generation evaluation from `SemanticIDGenerativeRecommender` into `SemanticIDEncoderDecoder`.
- [x] 2.3 Move only the active Lightning training shell behavior from `TransformerBaseModule` into `SemanticIDEncoderDecoder`.
- [x] 2.4 Remove `postprocessor` and `aggregator` constructor parameters and internal state from the TIGER model path.
- [x] 2.5 Ensure train-only dependencies such as `loss_function` and `evaluator` do not block `tiger_inference` model instantiation when unused.

## 3. Dead Code and Configuration Cleanup

- [x] 3.1 Remove `postprocessor: null` and `aggregator: null` from TIGER train/inference model configs.
- [x] 3.2 Delete or fully detach `src/recommendation/base_recommender.py` once no runtime references remain.
- [x] 3.3 Delete or fully detach `src/common/modules/transformer_base_module.py` once no runtime references remain.
- [x] 3.4 Search for stale `SemanticIDGenerativeRecommender`, `TransformerBaseModule`, `postprocessor`, `aggregator`, and model-side `codebooks` references and remove or update them.

## 4. Verification

- [x] 4.1 Run Hydra compose/instantiate smoke checks for `experiment=tiger_train` and `experiment=tiger_inference`.
- [x] 4.2 Run minimal `SemanticIDEncoderDecoder` train/eval/predict smoke checks covering forward/model_step/generation-facing paths.
- [x] 4.3 Verify model-side semantic ID tensor shape is `(num_items, num_hierarchies)` and prefix validation still works.
- [x] 4.4 Run targeted static checks for touched Python files.
- [x] 4.5 Run `openspec validate consolidate-tiger-generation-model --strict`.
