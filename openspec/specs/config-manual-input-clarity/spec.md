# config-manual-input-clarity Specification

## Purpose
TBD - created by archiving change clarify-config-manual-inputs. Update Purpose after archive.
## Requirements
### Requirement: Long config files SHALL expose manual inputs clearly
较长的 experiment 配置文件 SHALL 在清晰、统一的位置暴露用户需要手动填写的字段。

#### Scenario: User opens an experiment config
- **WHEN** 用户打开一个较长的 experiment 配置文件
- **THEN** 必须能在文件顶部附近找到所有主要手动输入字段
- **AND** 不需要在深层结构中再次寻找同类手填入口

### Requirement: Config files SHALL have consistent visual sections
主入口与较长 experiment 配置 SHALL 具有一致的 section 顺序与视觉分割方式。

#### Scenario: Reader scans a long config file
- **WHEN** 读者浏览一个较长配置文件
- **THEN** 文件必须使用统一的块注释和空行分隔主要 section

### Requirement: Deep placeholders SHALL not duplicate manual input entrypoints
深层配置中的裸 `???` 占位 SHALL NOT 再与 experiment 顶层手动输入字段形成重复入口。

#### Scenario: Semantic ID path is configured for recommendation experiment
- **WHEN** 用户通过顶层字段提供 `semantic_id_path`
- **THEN** 深层 semantic-id 相关配置必须引用该顶层字段
- **AND** 不得保留第二套裸 `???` 手填入口

### Requirement: Default path config SHALL defer to top-level inputs
默认路径配置中的手动输入字段 SHALL 透传顶层输入，而不是再次要求用户填写。

#### Scenario: Data directory is configured through experiment top-level field
- **WHEN** experiment 顶层提供 `data_dir`
- **THEN** `paths.default.data_dir` 必须透传该值

### Requirement: Inactive model implementation toggles SHALL stay out of config
Model configuration files SHALL NOT expose implementation toggles that are unused by all executable experiments and always set to the default no-op value.

#### Scenario: Quantization model config removes inactive CPU initialization toggle
- **WHEN** an implementation toggle such as `initialize_on_cpu` is not enabled by any executable experiment or script
- **THEN** the model config MUST omit that field
- **AND** users MUST NOT need to decide whether to set the inactive toggle

### Requirement: Quantization training configs SHALL hide internal loop hooks
Quantization training configuration files SHALL NOT expose internal training-loop hook fields that users are not expected to choose manually.

#### Scenario: User opens quantization training config
- **WHEN** 用户打开 RKMeans、RVQ 或 RQVAE training model config
- **THEN** config MUST NOT contain a `training_loop_function` field
- **AND** users MUST NOT need to understand manual optimization hooks to run quantization training

