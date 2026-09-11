## ADDED Requirements

### Requirement: Search-ranking diagnosis SHALL extend the existing unified pipeline optionally
系统 SHALL 用显式 `widened_recommendation_output_path` 和默认关闭的分析开关接入 paired analysis，继续通过根脚本、`src.main`、launcher、`Trainer.test` 执行。

#### Scenario: New analysis is enabled
- **WHEN** 用户启用 paired analysis
- **THEN** fixed/widened recommendation 与 trace MUST 完整存在，数据解析 MUST 归共享 Artifact resolver，统计 MUST 归 diagnosis domain
- **AND** 新脚本 flag MUST 支持等号/空格形式、拒绝空值，保留 required data-dir、默认 seed 42、notes、dry-run 与末尾 Hydra override 优先级

#### Scenario: Legacy invocation is used
- **WHEN** 新分析开关关闭且新输入为 null
- **THEN** 原无 trace、单 trace 和普通 diagnosis 调用 MUST 保持兼容

### Requirement: New analysis evidence SHALL have independently versioned outputs
启用分析时 evidence Artifact SHALL 包含 `search_ranking_by_user.csv`、`search_ranking_by_group.csv`、`prefix_attrition_by_layer.csv`、`static_risk_overlap.csv`、`static_risk_standardized_by_layer.csv`、`frequency_attrition_descriptives.csv` 中对应已启用模块的文件，并在 manifest/summary 中记录输入身份、版本、估计量与质量状态。

#### Scenario: Complete reanalysis evidence is published
- **WHEN** 两个新分析模块均成功完成
- **THEN** 六个新增文件 MUST 与原 evidence 通过共享 writer 一起发布，版本 MUST 包含 tail_search_ranking_v1 和 static_risk_overlap_v1
- **AND** 原始 URI、resolved path、split、分析配置和不可用原因 MUST 可审计

#### Scenario: Only one analysis module is enabled
- **WHEN** 另一个模块未启用或输入不可用
- **THEN** summary MUST 标明模块状态，不得生成伪造的全零结果表

### Requirement: Legacy statistical meanings SHALL not be silently overwritten
系统 SHALL 保留原始 raw damage、priority、CSV、summary 和 verdict 含义，对旧 prefix matched/CI 增加 legacy 说明，新分析结果 MUST 使用新字段与版本。

#### Scenario: Corrected estimate differs from legacy output
- **WHEN** 新标准化结果与旧 matched 指标方向不同
- **THEN** 二者 MUST 以不同名称和估计量说明保留，不能覆盖旧 run 或 Artifact
- **AND** 新分析 MUST 不自动将前置 H2 No-Go 改为 supported
