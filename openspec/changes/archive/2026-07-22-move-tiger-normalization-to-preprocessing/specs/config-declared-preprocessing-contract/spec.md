## MODIFIED Requirements

### Requirement: preprocessing chain SHALL be directly declared in config for readable experiment pipelines
在以可读性优先的实验 data 配置中，preprocessing chain SHALL 可以直接在配置文件中声明，并显式展示每一步 preprocessing 及其最小必要参数。配置声明的 preprocessing chain MUST support row-preserving, row-filtering, row-expanding, label-generating, and fixed-length sequence-normalizing preprocessing steps.

#### Scenario: rkmeans_train 配置显式声明 preprocessing chain
- **WHEN** 维护者查看 `rkmeans_train` 的 data 配置
- **THEN** 他 MUST 能直接看到 preprocessing 的顺序与每步参数，而不需要再跳转到 assembler 推导逻辑

#### Scenario: preprocessing 参数不通过 resolver 派生
- **WHEN** 维护者查看 preprocessing 的配置参数
- **THEN** 这些参数 MUST 以字面量或局部可见加载配置的形式出现
- **THEN** 它们 MUST NOT 依赖额外的 Hydra resolver 中间派生层

#### Scenario: dataset 直接消费已声明好的 preprocessing functions
- **WHEN** dataset 初始化 preprocessing chain
- **THEN** 它 MUST 直接读取 `dataset_config.preprocessing_functions`
- **THEN** 它 MUST NOT 再负责推导这些 preprocessing 的参数

#### Scenario: TIGER 配置显式声明扩展、label 和 normalize preprocessing
- **WHEN** 维护者查看 `tiger_train` 的 data 配置
- **THEN** train preprocessing chain MUST 显式声明 SID causal duplicate row expansion step
- **AND** train/eval preprocessing chains MUST 显式声明 TIGER label generation step
- **AND** train/eval preprocessing chains MUST 显式声明 `normalize_sequence` step after label generation
