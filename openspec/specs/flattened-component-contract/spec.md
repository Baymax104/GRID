# flattened-component-contract Specification

## Purpose
TBD - created by archiving change flatten-pseudo-reusable-components. Update Purpose after archive.
## Requirements
### Requirement: 伪复用组件 SHALL 展平为消费方文件内的函数
distance_functions、clustering_initializers、aggregation_strategy、quantization_strategies 这类仅提供多备选但实际只用一个实现的组件，SHALL 展平为消费方模型文件内的函数，不再以 ABC + 注入的模式存在。

#### Scenario: distance computation 作为函数内联到 quantization 模型
- **WHEN** 维护者查看 quantization 模型文件
- **THEN** 距离计算逻辑 MUST 作为文件内函数直接可见，不通过外部 distance_function 注入

#### Scenario: K-Means++ 初始化作为函数内联到 quantization 模型
- **WHEN** 维护者查看 quantization 模型的初始化逻辑
- **THEN** K-Means++ 初始化 MUST 作为文件内函数直接可见，不通过外部 initializer 注入

#### Scenario: STE 量化作为函数内联到 VQ 模型
- **WHEN** 维护者查看 VQ 模型的量化逻辑
- **THEN** STE 量化 MUST 作为文件内函数直接可见，不通过外部 quantization_strategy 注入

#### Scenario: Mean 聚合作为函数内联到 EmbeddingAggregator
- **WHEN** 维护者查看 EmbeddingAggregator
- **THEN** Mean 聚合逻辑 MUST 作为文件内函数直接可见，不通过外部 aggregation_strategy 注入

### Requirement: 展平后的配置 SHALL 不再注入展平组件
配置文件中 SHALL NOT 再出现 distance_function、initializer、quantization_strategy、aggregation_strategy 等展平组件的注入参数。

#### Scenario: quantization model config 无展平组件注入
- **WHEN** 维护者检查 quantization model 配置
- **THEN** 配置 MUST 不再包含 distance_function、initializer、quantization_strategy 的 `_target_` 引用

#### Scenario: sem_embeds model config 无 aggregation_strategy 注入
- **WHEN** 维护者检查 sem_embeds_inference model 配置
- **THEN** 配置 MUST 不再包含 aggregation_strategy 的 `_target_` 引用

### Requirement: common/components SHALL 不保留死代码
common/components 中保留的模块（model_output、loss_functions、scheduler）SHALL NOT 包含零引用的死类；已展平的模块文件和不再承载共享模块的 common modules 包 SHALL 删除。

#### Scenario: 保留模块无死类
- **WHEN** 维护者检查 loss_functions.py
- **THEN** 它 MUST 不包含零引用的类（FullBatchCrossEntropyLoss 已删除）

#### Scenario: 展平模块文件已删除
- **WHEN** 维护者检查 common/components 目录
- **THEN** distance_functions.py、clustering_initializers.py、aggregation_strategy.py、quantization_strategies.py、optimizer.py、eval_metrics.py MUST 不存在

#### Scenario: common modules 包已删除
- **WHEN** 维护者检查 `src/common`
- **THEN** `src/common/modules` MUST 不存在
- **THEN** 官方代码与配置 MUST NOT 引用 `src.common.modules.*`

