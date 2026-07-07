## ADDED Requirements

### Requirement: preprocessing 装配职责 SHALL 归属于 dataset 模块
在新的 data 架构中，preprocessing 的装配必须由 dataset 模块基于 `dataset_config` 内的声明配置完成，而不是由 preprocessing 反向读取 `dataset_config` 或依赖大量 YAML resolver 拼装运行时依赖。

#### Scenario: dataset 内部装配 preprocessing chain
- **WHEN** dataset 初始化并准备处理数据行
- **THEN** 它 MUST 基于 `dataset_config` 中的 preprocessing 声明配置装配出顺序执行的 preprocessing callables

#### Scenario: preprocessing 不反向读取 dataset_config
- **WHEN** 维护者检查 preprocessing 执行函数的调用路径
- **THEN** preprocessing MUST NOT 以参数形式接收 `dataset_config` 或其他模块级 config 对象

#### Scenario: dataset 是 preprocessing 的上游装配者
- **WHEN** 维护者检查 dataset 与 preprocessing 的依赖方向
- **THEN** 依赖关系 MUST 为 `dataset_config -> dataset -> preprocessing`
