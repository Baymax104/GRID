## 1. DataModule and config model cleanup

- [x] 1.1 Change the unified `BaseDataModule._build_collate_fn()` to return `curr_config.collate_fn` without `partial(...)` injection.
- [x] 1.2 Remove the now-unused `functools.partial` import from `src/data/datamodules/sequence.py`.
- [x] 1.3 Remove `labels`, `sequence_length`, `masking_token`, and `padding_token` from `SequenceDataloaderConfig`.
- [x] 1.4 Update `SequenceDataloaderConfig` docstring to describe only dataloader/runtime fields.

## 2. TIGER config migration

- [x] 2.1 Move `labels`, `sequence_length`, `masking_token`, and `padding_token` into `configs/data/tiger_train.yaml` `train_collate`.
- [x] 2.2 Move `labels`, `sequence_length`, `masking_token`, and `padding_token` into `configs/data/tiger_train.yaml` `eval_collate`.
- [x] 2.3 Remove those collate-only fields from TIGER train/val/test dataloader blocks.
- [x] 2.4 Move `sequence_length` and `padding_token` into `configs/data/tiger_inference.yaml` `collate`.
- [x] 2.5 Remove `labels`, `masking_token`, `sequence_length`, and `padding_token` from TIGER predict dataloader block.

## 3. Verification

- [x] 3.1 Search sequence data configs to confirm no `SequenceDataloaderConfig` block still declares `labels`, `sequence_length`, `masking_token`, or `padding_token`.
- [x] 3.2 Run `uv run python -m compileall -q src/data/datamodules/sequence.py src/data/components/config_models.py`.
- [x] 3.3 Run `uv run ruff check src/data/datamodules/sequence.py src/data/components/config_models.py`.
- [x] 3.4 Hydra compose and instantiate `experiment=tiger_train` and `experiment=tiger_inference` with a small semantic ID bundle.
- [x] 3.5 Smoke test `BaseDataModule._build_collate_fn()` returns the configured collate partial and collate calls still produce expected train/eval/inference batch shapes.
- [x] 3.6 Run `openspec validate flatten-sequence-collate-config --strict`.
