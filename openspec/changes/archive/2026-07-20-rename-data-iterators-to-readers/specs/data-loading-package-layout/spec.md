## MODIFIED Requirements

### Requirement: components 模块文件名 SHALL 准确反映内容职责
`src/data/components/` 下的模块文件名 SHALL 与其内容职责对应，并遵循复数、去下划线的统一命名风格。

#### Scenario: 原始文件数据读取器模块命名
- **WHEN** 维护者查找原始文件数据读取器
- **THEN** 这些定义 MUST 位于 `src/data/components/readers.py`
- **THEN** 该目录 MUST NOT 存在名为 `iterators.py` 的模块作为当前生效实现
