## Why

当前 `rkmeans_train` 的 data 链路虽然已经开始向新 data 架构迁移，但 preprocessing 仍存在两个明显问题：

1. `dataset_config` 中混入了大量仅供 preprocessing 使用的派生字段（如 `features_to_consider`、`field_type_map`、`embedding_map` 等）；
2. 这些派生字段大量依赖 Hydra 自定义 resolver 从 `features` 推导，导致配置阅读链路过长、语义分散，不利于后续迁移到其他实验。

用户进一步明确了新的架构约束：

- preprocessing 是 dataset 的子模块；
- dataset 接收 `dataset_config`；
- `dataset_config` 只描述 preprocessing 的声明式配置；
- dataset 在模块内部负责装配 preprocessing；
- preprocessing 本身不能再接收 `dataset_config` 或其他模块级 config 对象。

因此，本次需要把 preprocessing 的装配职责收回到 dataset 模块内部，并减少 YAML 中依赖 resolver 派生中间属性的做法。

## What Changes

- 将 `rkmeans_train` 的 preprocessing 装配收敛为 `dataset -> preprocessing` 的单向依赖关系
- 让 `dataset_config` 承载 preprocessing 的声明式配置，而不是承载大量 preprocessing 专用派生字段
- 由 dataset 内部根据 preprocessing 配置装配最小参数化的 preprocessing callables
- 逐步消除 preprocessing 函数对 `dataset_config` 的依赖
- 减少 `rkmeans_train.yaml` 中为 preprocessing 服务的自定义 resolver 派生属性

## Capabilities

### New Capabilities
- `dataset-owned-preprocessing-assembly`: 规定 preprocessing 的装配职责归属于 dataset，而不是外部 YAML resolver 或 preprocessing 对 `dataset_config` 的反向依赖
- `preprocessing-minimal-parameter-contract`: 规定 preprocessing 函数只能接收最小必要参数，而不是模块级 config 对象

### Modified Capabilities
- `data-reader-factory-contract`: 在稳定 reader factory 基础上，补充 dataset 对 preprocessing 装配的内部职责边界

## Impact

- 受影响代码：`src/data/components/datasets.py`、`src/data/components/preprocessing.py`、必要时新增 preprocessing assembler/helper
- 受影响配置：`configs/data/rkmeans_train.yaml`、相关 dataset config dataclass
- 受影响范围：先以 `rkmeans_train` 为模板，不要求本次同时迁完全部实验
- 预期收益：data 配置更易读，preprocessing 更纯、更局部，后续实验迁移成本更低
