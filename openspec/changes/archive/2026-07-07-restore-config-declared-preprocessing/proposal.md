## Why

当前 `rkmeans_train` 的 preprocessing 虽然已经纯化为 row-only 且不再接收 `dataset_config`，但其装配方式又收敛到了 dataset 内部的 assembler / registry 逻辑。这种方式在扩展性上更强，但对当前场景并不最优：

- preprocessing 通常只在单个实验配置中声明一次；
- 用户当前优先级是“可读性优先”，希望一眼看出做了哪些 preprocessing；
- dataset 内部 assembler 与 `preprocessing_steps`、`features` 这条中间层增加了阅读跳转成本。

因此，本次希望恢复到“配置文件直接声明 preprocessing chain”的方式，但保留新架构已经获得的收益：

- preprocessing 函数只接收最小必要参数；
- 不再接收 `dataset_config`；
- 不再依赖 Hydra resolver 派生 preprocessing 参数。

## What Changes

- 将 `rkmeans_train` 的 preprocessing 构建方式恢复为由 YAML 配置直接声明 `preprocessing_functions`
- 每个 preprocessing 的参数直接以字面量或局部可见加载配置写在 YAML 中
- 删除 `rkmeans_train` 主链路对 `preprocessing_assembly.py`、`preprocessing_steps`、`features` 派生层的依赖
- 保持 dataset 只消费已声明好的 preprocessing callables，不再负责推导 preprocessing 参数

## Capabilities

### New Capabilities
- `config-declared-preprocessing-contract`: 规定 preprocessing chain 可以直接在 experiment data 配置中显式声明，参数以可读性优先的字面量形式提供

### Modified Capabilities
- `dataset-owned-preprocessing-assembly`: 对 `rkmeans_train` 主链路收敛为“dataset 消费 preprocessing config”，而不是“dataset 内部推导 preprocessing 参数”

## Impact

- 受影响代码：`src/data/components/datasets.py`、`src/data/components/config_models.py`、必要时删除 `src/data/components/preprocessing_assembly.py`
- 受影响配置：`configs/data/rkmeans_train.yaml`
- 目标是让 `rkmeans_train` 的 preprocessing 声明更直接可读，同时不回退到旧的 resolver / dataset_config 反向依赖模式
