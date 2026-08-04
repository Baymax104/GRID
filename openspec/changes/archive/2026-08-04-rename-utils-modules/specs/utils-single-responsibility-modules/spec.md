## MODIFIED Requirements

### Requirement: utils 包中每个模块 SHALL 保持单一职责
`src/utils/` 包中的每个模块 SHALL 只承载一类职责。禁止以 `utils.py` 或类似泛化命名的 catch-all 模块堆积不相关功能。模块文件名 SHALL 使用职责名，避免冗余 `_utils` 后缀。

#### Scenario: 新增工具函数时选择正确模块
- **WHEN** 维护者需要在 utils 包中新增一个工具函数
- **THEN** 该函数 MUST 放入已存在的、职责匹配的模块
- **THEN** 若无匹配模块，MUST 新建一个职责命名的模块（如 `extra.py`、`model.py`）
- **THEN** 该函数 MUST NOT 放入一个名称泛化的 catch-all 模块
- **THEN** 新模块名 MUST NOT 使用冗余 `_utils` 后缀

#### Scenario: 启动预处理与模型操作分离
- **WHEN** 维护者检查 utils 包结构
- **THEN** 启动预处理函数（如 `extras`）MUST 位于 `extra.py`
- **THEN** 模型模块操作函数（如 `delete_module`、`reset_parameters`）MUST 位于 `model.py`
- **THEN** 这两类函数 MUST NOT 共存于同一个模块

### Requirement: 跨域业务逻辑 SHALL 移至对应域而非留在 utils 中
当 utils 中的函数仅被单一业务域消费且依赖该域的概念时，该函数 SHALL 移至对应域的模块中。

#### Scenario: 分词辅助函数归属 data 域
- **WHEN** 维护者检查 `load_tokenize` 的位置
- **THEN** 它 MUST 位于 `src/data/components/tokenization.py` 或 data 域内的其他模块
- **THEN** 它 MUST NOT 位于 `src/utils/` 中

#### Scenario: 序列掩码工具归属模型工具域
- **WHEN** 维护者检查 `create_last_k_mask` 的位置
- **THEN** 它 MUST 位于 `src/utils/model.py`
- **THEN** 它 MUST NOT 位于 `src/inference/utils.py`（该模块专用于 keyed prediction bundle 协议）
