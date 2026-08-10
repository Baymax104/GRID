## Why

`ItemTextBatch` 当前只有定义没有运行引用；同时 `ModelOutput` 和 `src.inference.utils` 中的 keyed prediction bundle 加载/查询函数已经跨 embedding、quantization、recommendation、data preprocessing 使用。继续放在 `src.inference` 会把数据产物契约误归类为推理回调实现细节。

本变更将 keyed prediction bundle 的数据容器与加载查询工具收敛到 data 领域，使 preprocessing 和配置入口使用同一组 data utilities。

## What Changes

- 删除未使用的 `ItemTextBatch`。
- 将 `ModelOutput` 从 `src/inference/model_output.py` 移入 `src/data/components/data_models.py`。
- 将 `src/inference/utils.py` 中的 `load_model_output`、`load_semantic_id_tensor`、`gather_predictions_by_keys` 移入 `src/data/utils.py`。
- 更新 Python imports、Hydra `_target_` 配置和相关测试。
- 删除迁移后的空文件或旧入口，不保留 `src.inference.model_output` / `src.inference.utils` 兼容 re-export。
- 更新 OpenSpec living specs，使数据模型和 keyed bundle 访问路径与当前目录职责一致。

## Capabilities

### New Capabilities

### Modified Capabilities
- `data-model-role-separation`: 运行时 data model 集合删除 `ItemTextBatch`，并新增 `ModelOutput` 作为 keyed prediction bundle 数据容器。
- `prediction-output-protocol`: `ModelOutput` 的归属路径从 `src.inference.model_output` 改为 `src.data.components.data_models`。
- `keyed-prediction-bundle-artifact`: keyed prediction bundle 加载与查询函数从 `src.inference.utils` 改为 `src.data.utils`。

## Impact

- 影响代码：`src/data/components/data_models.py`、`src/data/utils.py`、`src/inference/`、embedding / quantization / recommendation 的 `ModelOutput` imports、data preprocessing imports。
- 影响配置：`configs/data/*.yaml` 与 TIGER model configs 中引用 `src.inference.utils.*` 的 `_target_`。
- 影响测试：prediction writer 测试、bundle 加载/查询相关测试或新增路径断言。
- 不新增依赖，不改变 keyed prediction bundle 文件格式：仍为 `{"keys": ..., "predictions": ...}`。
