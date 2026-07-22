## 1. 改造 label function 实现

- [x] 1.1 删除 `src/data/components/label_functions.py` 中的 `LabelFunction` 抽象基类及 `ABC` / `abstractmethod` import
- [x] 1.2 将 `Identity.transform_label` 迁移为纯函数 `identity_label(...)`，保持输出语义不变
- [x] 1.3 将 `NextKTokenMasking.transform_label` 迁移为纯函数 `next_k_token_masking(..., next_k=5)`，保持 masking、labels、label_location 语义和 shape 不变
- [x] 1.4 删除 `Identity` 和 `NextKTokenMasking` 类定义及类式 docstring

## 2. 更新 collate 与配置

- [x] 2.1 更新 `collate_fn_train`，直接调用 `label_generate_functions[field_name](sequence=..., padding_token=..., masking_token=...)`
- [x] 2.2 更新 `collate.py` 中 labels 类型注解和相关注释，移除 `.transform.transform_label` 协议描述
- [x] 2.3 更新 `configs/data/tiger_train.yaml` 的 `label_functions`，直接使用 `next_k_token_masking` 的 Hydra `_partial_` callable
- [x] 2.4 确认配置中不再存在 label `transform:` wrapper 或 `NextKTokenMasking` / `Identity` target

## 3. 同步 living specs

- [x] 3.1 新增 `openspec/specs/pure-label-function-contract/spec.md` 或确保归档 delta 能生成该 spec
- [x] 3.2 更新 `openspec/specs/tiger-sequence-data-contract/spec.md`，记录 TIGER label generator 直接使用纯函数 callable

## 4. 验证

- [x] 4.1 运行 grep 确认非 archive 的 Python/YAML/living specs 中无 `LabelFunction`、`NextKTokenMasking`、`Identity` 类协议残留引用
- [x] 4.2 运行 `uv run python -m compileall -q src/data/components/label_functions.py src/data/components/collate.py`
- [x] 4.3 运行 `uv run ruff check src/data/components/label_functions.py src/data/components/collate.py`
- [x] 4.4 执行 `collate_fn_train` smoke，确认 `next_k_token_masking` 的 `labels` 和 `label_location` shape 与旧语义一致
- [x] 4.5 执行 `tiger_train` Hydra compose + datamodule instantiate smoke，确认 label callable 可实例化并可被 collate 调用
- [x] 4.6 运行 `openspec validate replace-label-function-classes-with-functions --strict`
- [x] 4.7 运行 `openspec validate --specs --no-interactive`
- [x] 4.8 运行 `git diff --check`
