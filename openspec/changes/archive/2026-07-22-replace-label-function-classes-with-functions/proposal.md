## Why

当前 label generation 通过 `LabelFunction` 抽象类、`NextKTokenMasking` 类实例以及 `.transform.transform_label(...)` 间接调用完成，但这些对象没有状态化生命周期需求，实际只是对 tensor 做确定性转换。改为纯函数可以减少配置层级、删除不必要的继承接口，并让 collate 中的 label callable 来源更直接。

## What Changes

- 将 `src/data/components/label_functions.py` 中的 label generation 从类式协议改为纯函数协议。
- 删除 `LabelFunction` 抽象基类、`Identity` 类和 `NextKTokenMasking` 类。
- 新增等价纯函数 `next_k_token_masking(...)`，返回 `GeneratedLabels(input_ids, target_ids)` 并保持 `NextKTokenMasking` 语义不变。
- 修改 `collate_fn_train`，直接调用 `label_generate_functions[field_name](sequence=..., padding_token=..., masking_token=...)`，不再访问 `.transform.transform_label(...)`。
- 更新 `configs/data/tiger_train.yaml` 的 label 配置，使 Hydra 直接提供 `_partial_` 纯函数 callable。
- **BREAKING**：不再支持通过 Hydra `_target_` 实例化 `NextKTokenMasking` / `Identity` 类，也不再支持 label config 中的 `transform:` 包装层。

## Capabilities

### New Capabilities

- `pure-label-function-contract`: 定义 label generation 必须以纯函数 callable 暴露，并保持 label 输出协议稳定。

### Modified Capabilities

- `tiger-sequence-data-contract`: TIGER train/eval label 配置改为直接声明纯函数 callable，不再使用类实例和 `transform` wrapper。

## Impact

- `src/data/components/label_functions.py`：删除类协议，改为纯函数。
- `src/data/components/collate.py`：更新 label callable 调用方式与类型注解。
- `configs/data/tiger_train.yaml`：迁移 `label_functions` 配置。
- `openspec/specs/`：新增纯函数 label contract，并更新 TIGER sequence data contract。
- 验证重点：`collate_fn_train` label shape smoke、TIGER Hydra compose/instantiate、`compileall`、`ruff`、OpenSpec strict validation。
