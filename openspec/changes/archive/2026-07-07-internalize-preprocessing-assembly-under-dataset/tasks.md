## 1. 设计 preprocessing 声明配置与装配入口

- [x] 1.1 明确 `rkmeans_train` 当前 preprocessing 所需的最小参数集合
- [x] 1.2 为 dataset_config 设计仅包含“preprocessing 声明配置”的结构，避免继续承载 preprocessing 专用派生字段
- [x] 1.3 在 dataset 模块内部定义 preprocessing assembler/helper 的入口与职责边界

## 2. 收回 preprocessing 装配职责到 dataset

- [x] 2.1 更新 `src/data/components/datasets.py`，由 dataset 基于 `dataset_config` 装配 preprocessing callables
- [x] 2.2 确保 preprocessing 装配后的执行路径仍是 `list[callable]` 顺序链路
- [x] 2.3 避免 preprocessing 装配职责继续散落在 YAML resolver 表达式中

## 3. 纯化 preprocessing 函数接口

- [x] 3.1 更新 `rkmeans_train` 当前实际使用的 preprocessing 函数签名，移除 `dataset_config` / `config` 参数
- [x] 3.2 让这些 preprocessing 只接收最小必要参数（如 `features_to_consider`、`field_type_map`、`embedding_bundle` 等）
- [x] 3.3 确保 preprocessing 不再反向依赖 dataset 模块级对象

## 4. 简化 rkmeans_train 配置

- [x] 4.1 清理 `configs/data/rkmeans_train.yaml` 中仅服务 preprocessing 的 resolver 派生字段
- [x] 4.2 保留 `features` 作为源描述，但将派生逻辑尽可能迁到 Python 侧 assembler/helper
- [x] 4.3 让 `dataset_config` 回归 dataset-level 语义，避免成为 preprocessing metadata 容器

## 5. 验证收尾

- [x] 5.1 grep 验证 `rkmeans_train` 主链路中的 preprocessing 不再接收 `dataset_config`
- [x] 5.2 import / instantiate smoke check：dataset 能根据 `dataset_config` 成功装配 preprocessing chain
- [x] 5.3 Hydra compose 最小验证：`experiment=rkmeans_train` 配置可解析且较当前减少 resolver 派生中间属性
- [x] 5.4 总结该模式对后续实验迁移的可复用规则
