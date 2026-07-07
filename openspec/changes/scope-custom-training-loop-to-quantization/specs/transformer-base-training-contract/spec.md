## ADDED Requirements

### Requirement: 通用 transformer 训练基类 SHALL 使用标准 automatic optimization 契约
项目中的通用 transformer 训练基类不得继续暴露 quantization 特有的手动训练 loop 注入接口。

#### Scenario: TransformerBaseModule 不再暴露 custom training loop hook
- **WHEN** 维护者查看 `TransformerBaseModule` 的构造参数与 `training_step()` 逻辑
- **THEN** 该基类 MUST NOT 暴露 `training_loop_function` 这类 quantization 特有 hook
- **THEN** 该基类的训练流程 MUST 回到标准 Lightning automatic optimization 契约

#### Scenario: Recommendation 主链配置不再暴露该机制
- **WHEN** 维护者检查 recommendation / transformer 训练配置
- **THEN** 这些配置 MUST NOT 再出现 `training_loop_function` 暴露项
