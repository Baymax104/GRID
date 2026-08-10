## MODIFIED Requirements

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
