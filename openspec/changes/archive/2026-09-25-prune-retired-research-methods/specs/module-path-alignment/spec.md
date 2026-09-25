## ADDED Requirements

### Requirement: Recommendation directories SHALL correspond to active method modules
`src/recommendation/` 下的顶层方法目录 SHALL 对应当前可执行方法；已结题研究方法的实现 SHALL 不以独立目录、adapter 或兼容壳继续存在。

#### Scenario: Inspect recommendation layout after pruning
- **WHEN** 清理完成后枚举 recommendation 方法目录
- **THEN** 方法目录只包含 `tiger` 与 `liger`
- **AND** 共享运行逻辑继续通过既有 `src.common`、`src.data` 和 `src.utils` seam 提供
