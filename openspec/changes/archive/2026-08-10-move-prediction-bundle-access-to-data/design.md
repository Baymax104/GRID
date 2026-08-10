## Context

当前数据相关对象分布不一致：

- `ItemTextBatch` 位于 `src/data/components/data_models.py`，但只在定义处出现，没有运行引用。
- `ModelOutput` 位于 `src/inference/model_output.py`，但由 embedding、quantization、recommendation、prediction writer、data preprocessing 共同使用。
- `src/inference/utils.py` 提供 keyed prediction bundle 的加载、排序、重复 key 校验、semantic ID tensor 提取和按 key 查询；其中 `gather_predictions_by_keys` 直接被 `src/data/components/preprocessing.py` 使用，多个 `configs/data/*.yaml` 也通过 `_target_` 引用 `load_model_output`。

这些函数的接口不是推理 callback 的接口，而是数据产物访问接口。将它们迁入 `src/data/utils.py` 能让 preprocessing、data configs 和 bundle 查询协议的归属保持一致。

## Goals / Non-Goals

**Goals:**

- 删除没有调用方的 `ItemTextBatch`。
- 将 `ModelOutput` 迁入 `src/data/components/data_models.py`。
- 将 `load_model_output`、`load_semantic_id_tensor`、`gather_predictions_by_keys` 迁入 `src/data/utils.py`。
- 更新所有 Python imports 与 Hydra `_target_`。
- 删除旧的 `src/inference/model_output.py` 与 `src/inference/utils.py`。
- 更新 OpenSpec living specs 和 focused tests。

**Non-Goals:**

- 不改变 `ModelOutput` 的字段或构造方式。
- 不改变 keyed prediction bundle 的文件格式。
- 不改变 `LocalPickleWriter` 的 callback 职责或落盘行为。
- 不迁移 `src/inference/prediction_writers.py` 或 `src/inference/postprocessing.py` 到 data。
- 不新增兼容 re-export。

## Decisions

### 删除 ItemTextBatch

`ItemTextBatch` 当前只在 `data_models.py` 定义，没有代码、配置或测试引用。删除它可以减少 data model 的误导性接口。

替代方案是保留以备未来使用；拒绝原因是 living code 中没有消费者，保留会扩大 data model 的公共表面。

### ModelOutput 移入 data_models

`ModelOutput` 是 keyed prediction bundle 的运行时容器，描述 `keys` 与 `predictions` 的数据契约。它虽然由 inference writer 消费，但并不属于 writer 实现；模型 `predict_step` 产出、writer 缓存、preprocessing 查询都共享这一个数据接口。

放入 `src/data/components/data_models.py` 后，所有运行时 batch / model input / keyed prediction container 都在 data model 模块中定义。

### inference/utils 函数移入 data/utils

`load_model_output`、`load_semantic_id_tensor`、`gather_predictions_by_keys` 是数据产物的加载与查询逻辑。尤其 `gather_predictions_by_keys` 是 preprocessing 的核心 helper，与现有 `src/data/utils.py` 中的数据处理工具职责一致。

本轮按用户明确要求迁入 `src/data/utils.py`，不额外创建 `src/data/prediction_bundles.py`。

### 不保留旧 inference 入口

迁移后不保留 `src.inference.model_output` 或 `src.inference.utils` 的兼容 re-export。

理由：

- 项目内部路径可以一次性更新，并通过 residual scan 捕获遗漏。
- 兼容层会让旧归属继续存在，削弱这次目录结构调整的效果。

## Risks / Trade-offs

- [Risk] 外部脚本仍引用 `src.inference.utils.*` 或 `src.inference.model_output.ModelOutput`。  
  Mitigation: 官方代码、配置、测试全部迁移；旧入口直接失败以暴露外部残留。

- [Risk] `src/data/utils.py` 职责继续扩大。  
  Mitigation: 本次只迁入 data preprocessing 已经依赖的 bundle 访问函数；如后续继续增长，再拆分为 `src/data/prediction_bundles.py`。

- [Risk] Hydra `_target_` 漏改会运行时失败。  
  Mitigation: 对 `src.inference.utils`、`src.inference.model_output` 做 residual scan，并运行 focused config/import tests。

## Migration Plan

1. 从 `data_models.py` 删除 `ItemTextBatch`，新增 `ModelOutput`。
2. 将 `src/inference/utils.py` 函数移动到 `src/data/utils.py`，更新内部 import。
3. 更新所有 Python imports。
4. 更新所有 Hydra `_target_` 配置。
5. 删除旧 `src/inference/model_output.py` 与 `src/inference/utils.py`。
6. 更新 living specs。
7. 增加或调整 focused tests。
8. 运行 residual scans、focused pytest、scoped ruff、OpenSpec strict validation。

## Open Questions

无。
