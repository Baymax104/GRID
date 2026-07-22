## 1. Collate 入口合并

- [x] 1.1 新增 `sample_sid_causal_duplicate_sequences` helper，并迁移现有 SID causal duplicate 采样逻辑
- [x] 1.2 为 `collate_fn_train` 增加 `enable_sid_causal_duplicate`、`sequence_field_name`、`sid_hierarchy`、`max_batch_size` 参数
- [x] 1.3 在 `collate_fn_train` 内部按开关调用 helper，并在缺少必要增强参数时抛出清晰错误
- [x] 1.4 删除 `collate_with_sid_causal_duplicate` public wrapper

## 2. 配置迁移

- [x] 2.1 将 `configs/data/tiger_train.yaml` 的 `train_collate._target_` 改为 `collate_fn_train`
- [x] 2.2 在 `train_collate` 上配置 `enable_sid_causal_duplicate: true`、`sequence_field_name`、`sid_hierarchy`、`max_batch_size`
- [x] 2.3 确认 `eval_collate` 继续使用 `collate_fn_train` 且不启用 SID causal duplicate augmentation

## 3. Specs 同步

- [x] 3.1 新增 living spec `unified-tiger-train-collate`
- [x] 3.2 更新 `tiger-specific-batch-contract` 和 `tiger-sequence-data-contract` living specs
- [x] 3.3 检查非 archive 的代码、配置、spec 中不存在 `collate_with_sid_causal_duplicate` 残留

## 4. 验证

- [x] 4.1 运行 `uv run python -m compileall -q src/data/components/collate.py`
- [x] 4.2 运行 `uv run ruff check src/data/components/collate.py`
- [x] 4.3 运行 helper smoke，验证连续子序列采样语义和 `max_batch_size`
- [x] 4.4 运行 `collate_fn_train` augmentation enabled smoke，验证输出仍为 `TigerModelInput` / `TigerLabelData`
- [x] 4.5 运行 `collate_fn_train` augmentation disabled smoke，验证 eval/test 行为不变
- [x] 4.6 运行 `tiger_train` Hydra compose + datamodule instantiate smoke
- [x] 4.7 运行 `openspec validate unify-tiger-train-collate --strict`
- [x] 4.8 运行 `openspec validate --specs --no-interactive`
- [x] 4.9 运行 `git diff --check`
