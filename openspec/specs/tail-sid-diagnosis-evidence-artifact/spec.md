# tail-sid-diagnosis-evidence-artifact Specification

## Purpose
TBD - created by archiving change strengthen-tail-sid-diagnosis-evidence. Update Purpose after archive.
## Requirements
### Requirement: Diagnosis evidence SHALL be written as a stable structured output
The system SHALL write each completed official Tail-SID diagnosis to a dedicated local analysis directory using JSON and CSV files with stable schemas. The output SHALL contain run-level summary and verdict data plus group-, item-, prefix-, and bounded harmful-pair evidence needed for audit and downstream methods.

#### Scenario: Complete structural diagnosis writes required files
- **WHEN** Tail-SID diagnosis completes without recommendation outcomes
- **THEN** the output directory MUST contain `summary.json`, `group_metrics.csv`, `item_damage_scores.csv`, `prefix_risk_scores.csv`, and `harmful_overlap_pairs.csv`
- **AND** `summary.json` MUST identify recommendation evidence as unavailable

#### Scenario: Recommendation evidence extends item outputs
- **WHEN** a recommendation output bundle is provided
- **THEN** item evidence MUST include label support and configured hit/rank outcome fields
- **AND** summary output MUST include the configured raw-damage correlation and frequency-matched comparison results

### Requirement: Diagnosis evidence schemas SHALL preserve raw and prioritized meanings
Evidence files SHALL use distinct fields for observed raw risk and frequency-aware priority. Item and prefix schemas MUST retain keyed IDs and the raw/model SID distinction so consumers do not infer row-index identity or mistake de-duplication for raw quantizer structure.

#### Scenario: Item evidence is keyed and auditable
- **WHEN** an item evidence row is written
- **THEN** it MUST include `item_id`, frequency group, training frequency, raw SID, model SID, de-duplication digit, raw component values, `raw_damage`, and `priority_score`
- **AND** the row order MUST NOT define item identity

#### Scenario: Prefix evidence supports downstream repair
- **WHEN** a prefix evidence row is written
- **THEN** it MUST include prefix depth/value, bucket size and group composition, suffix uniqueness, raw risk aggregates, semantic evidence availability, and separately named raw and priority prefix scores

### Requirement: Harmful-pair output SHALL be deterministic and bounded
The diagnosis SHALL bound harmful-pair output by configured global and/or per-item limits while preserving deterministic selection. Metadata MUST record the limits, sampling seed, selection rule, and whether any rows were truncated.

#### Scenario: Large pair sets cannot create unbounded output
- **WHEN** candidate harmful pairs exceed configured limits
- **THEN** the writer MUST retain pairs using the configured deterministic ranking and per-item/global limits
- **AND** summary metadata MUST report candidate, retained, and truncated counts

### Requirement: Diagnosis evidence SHALL be publishable as one W&B Artifact
When a W&B logger run is configured, the official diagnosis SHALL publish the complete local evidence directory as one Artifact through the logger-owned run. Publishing MUST NOT initialize, finish, or replace the logger run, and local-only output MUST remain usable without W&B.

#### Scenario: Logger-owned run publishes complete evidence
- **WHEN** diagnosis completes with an active W&B logger run
- **THEN** one Artifact of the configured diagnosis evidence type MUST contain every required evidence file
- **AND** Artifact metadata MUST record task, dataset, tokenizer input references, schema version, primary analysis settings, and verdict

#### Scenario: Local-only output does not require W&B
- **WHEN** no W&B logger is configured and local output is enabled
- **THEN** the complete evidence directory MUST still be written
- **AND** the writer MUST NOT initialize a W&B run

### Requirement: Diagnosis evidence output SHALL be atomic and fail visibly
The writer SHALL stage files before exposing a completed local output and SHALL propagate serialization or Artifact publication failures. A failed output MUST NOT be presented as a complete evidence Artifact.

#### Scenario: Serialization failure does not publish partial evidence
- **WHEN** a required evidence file cannot be serialized
- **THEN** the diagnosis run MUST fail before Artifact publication
- **AND** no completion marker or successful verdict file MUST be exposed

#### Scenario: Artifact upload failure is not swallowed
- **WHEN** W&B Artifact publication is configured and upload fails
- **THEN** the failure MUST propagate to the official diagnosis run
- **AND** the logger-owned run lifecycle MUST remain owned by the configured logger

### Requirement: Diagnosis evidence SHALL preserve module ownership and dependency direction
The evidence implementation SHALL keep input resolution and keyed assembly in the data layer, Tail-SID formulas and verdicts in the diagnosis domain, scalar lifecycle in the shared metric runtime, structured serialization/publication in common writers, and input lineage in the existing lineage callback. Shared infrastructure MUST consume domain-neutral protocols and MUST NOT import the Tail-SID diagnosis package.

#### Scenario: Data assembly does not compute diagnosis evidence
- **WHEN** semantic ID, embedding, testing label, or recommendation inputs are loaded and aligned
- **THEN** data-layer components MUST return keyed raw/runtime data without computing damage, statistical evidence, or verdicts
- **AND** W&B input resolution MUST continue through the shared Artifact resolver

#### Scenario: Common writer is domain neutral
- **WHEN** diagnosis evidence is handed to a common structured writer
- **THEN** the diagnosis domain MUST first adapt it to a generic named-document/named-table payload
- **AND** the common writer MUST NOT import diagnosis evidence types or branch on Tail-SID field names

#### Scenario: Existing lifecycle owners remain unchanged
- **WHEN** the official diagnosis is assembled and executed
- **THEN** the launcher and metric callback MUST remain free of diagnosis-specific conditional paths
- **AND** input `use_artifact` calls MUST remain owned by `WandbArtifactLineageCallback`
- **AND** run initialization/finalization MUST remain owned by configured loggers

#### Scenario: Unified entrypoint and Hydra output ownership are preserved
- **WHEN** diagnosis evidence is produced
- **THEN** execution MUST continue through the root script, `src.main`, unified launcher, and `Trainer.test`
- **AND** local output paths MUST derive from configured `paths.output_dir`

### Requirement: Tail-SID diagnosis SHALL optionally consume Prefix Trace Artifacts
Tail-SID diagnosis SHALL accept an optional fixed-beam Prefix Trace Artifact and an optional widened-beam Prefix Trace Artifact through explicit configuration fields and the shared Artifact resolver. Trace inputs SHALL be joined to recommendation labels and item evidence by business keys, not row positions.

#### Scenario: Fixed-beam trace is provided
- **WHEN** diagnosis receives a valid fixed-beam Prefix Trace Artifact
- **THEN** it MUST validate user keys, target labels, semantic-ID reference, data split and schema version
- **AND** it MUST compute layer-wise target probability/rank, survival and first-failure evidence

#### Scenario: Widened-beam trace is also provided
- **WHEN** diagnosis receives compatible fixed and widened trace Artifacts
- **THEN** it MUST validate checkpoint, split, labels and semantic-ID identity across them
- **AND** it MUST compute target-path recovery and failure-depth shift evidence

#### Scenario: Trace inputs are absent
- **WHEN** diagnosis runs without Prefix Trace Artifacts
- **THEN** all existing static and recommendation evidence behavior MUST remain available
- **AND** prefix-survival mechanism evidence MUST be marked unavailable

### Requirement: Diagnosis SHALL emit stable prefix-survival mechanism evidence
When trace evidence is available, diagnosis SHALL extend its structured output with stable layer/group survival, first-failure, competition-association and optional widened-beam recovery files plus a separately named mechanism verdict.

#### Scenario: Mechanism evidence is written
- **WHEN** fixed-beam trace analysis completes
- **THEN** the evidence directory MUST include layer-wise group metrics and first-failure metrics
- **AND** `summary.json` MUST report trace schema, split, beam width and mechanism-evidence availability

#### Scenario: Recovery evidence is written
- **WHEN** compatible widened-beam trace is available
- **THEN** the evidence directory MUST include widened-beam recovery metrics by popularity group and layer
- **AND** summary MUST distinguish recovery evidence from ordinary recommendation outcome metrics

### Requirement: Mechanism evidence SHALL preserve exploratory and confirmatory boundaries
Diagnosis SHALL record whether trace inputs came from evaluation or testing. Outputs intended as calibration statistics MUST reject testing input, and the H2 verdict SHALL remain distinct from existing structural and generation-risk verdicts.

#### Scenario: Evaluation trace is used for method development
- **WHEN** diagnosis is configured to emit calibration-statistics-ready evidence
- **THEN** every trace input MUST declare `data_split=evaluation`
- **AND** summary metadata MUST identify the output as development evidence

#### Scenario: Testing trace is presented as calibration statistics
- **WHEN** a testing trace is supplied to a calibration-statistics-ready analysis
- **THEN** diagnosis MUST fail with a data leakage error

#### Scenario: H2 evidence is unavailable or unsupported
- **WHEN** layer-wise survival or recovery evidence does not meet configured availability or stability rules
- **THEN** diagnosis MUST NOT change `raw_damage` or reuse `generation_risk_validity` as the H2 verdict
- **AND** the separately named mechanism verdict MUST report unavailable or not supported

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
