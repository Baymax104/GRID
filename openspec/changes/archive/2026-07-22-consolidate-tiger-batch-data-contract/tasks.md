## 1. 数据模型收敛

- [x] 1.1 在 `src/data/components/data_models.py` 中用 `GeneratedLabels`、`TigerModelInput`、`TigerLabelData` 替换旧 sequential/label output dataclasses
- [x] 1.2 删除 `SequentialModelInputData`、`SequentialModuleLabelData`、`LabelFunctionOutput` 类定义及相关导出
- [x] 1.3 确保 `TigerModelInput` 字段为 `input_ids`、`attention_mask`、`output_keys`
- [x] 1.4 确保 `TigerLabelData` 和 `GeneratedLabels` 只暴露 TIGER 需要的 target/input 字段

## 2. Label function 与 collate 迁移

- [x] 2.1 将 `next_k_token_masking` 改为返回 `GeneratedLabels(input_ids, target_ids)`
- [x] 2.2 将 `next_k_token_masking.target_ids` shape 改为 `(batch_size, next_k)`，并保持现有 masking 语义
- [x] 2.3 删除未使用的 `identity_label`
- [x] 2.4 将 `collate_fn_train` 改为返回 `(TigerModelInput, TigerLabelData)`
- [x] 2.5 将 `collate_fn_inference_for_sequence` 改为返回 `TigerModelInput`，并把 id 字段保存为 `output_keys`
- [x] 2.6 确认 `collate_with_sid_causal_duplicate` 继续兼容 dict batch 并调用新的训练 collate contract

## 3. TIGER 模型与配置迁移

- [x] 3.1 从 `SemanticIDEncoderDecoder.__init__` 和模型状态中删除 `feature_to_model_input_map`
- [x] 3.2 更新 `model_step`、`training_step`、`eval_step`、`validation_step`、`test_step`、`predict_step` 以使用 `TigerModelInput` / `TigerLabelData`
- [x] 3.3 更新推理输出逻辑，使 `ModelOutput.keys` 来源于 `TigerModelInput.output_keys`
- [x] 3.4 从 `configs/model/tiger_train.yaml` 和 `configs/model/tiger_inference.yaml` 删除 `feature_to_model_input_map`

## 4. Specs 与文档同步

- [x] 4.1 新增 living spec `tiger-specific-batch-contract`
- [x] 4.2 更新 `pure-label-function-contract`、`tiger-sequence-data-contract`、`data-model-role-separation`、`self-contained-tiger-generation-model` living specs
- [x] 4.3 检查非 archive 的代码、配置、spec 中不存在旧类名和旧字段名的非规范性残留

## 5. 验证

- [x] 5.1 运行相关 Python 文件 compileall
- [x] 5.2 运行相关 Python 文件 ruff check
- [x] 5.3 运行 train collate smoke，验证 `TigerModelInput.input_ids/attention_mask` 与 `TigerLabelData.target_ids` shape
- [x] 5.4 运行 inference collate smoke，验证 `output_keys` 与 `input_ids` 分离
- [x] 5.5 运行 TIGER `model_step` / `predict_step` smoke
- [x] 5.6 运行 `tiger_train` 和 `tiger_inference` Hydra compose + instantiate smoke
- [x] 5.7 运行 `openspec validate consolidate-tiger-batch-data-contract --strict`
- [x] 5.8 运行 `openspec validate --specs --no-interactive`
- [x] 5.9 运行 `git diff --check`
