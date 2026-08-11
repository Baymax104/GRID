## Why

最新 Tail-SID diagnosis 运行结果暴露出两个可用性问题：`damage` 分数因 robust z-score 的 IQR 接近 0 被放大到 `1e7+` 量级，且 stdout 使用手写 Markdown 表格，不利于快速判断结果好坏。需要先稳定指标量级，并用更清晰的表格展示核心结果。

## What Changes

- 稳定 `damage` / `tail_damage` / `prefix_risk` 的归一化逻辑，避免 near-constant 指标被 `eps` 放大。
- 在输出中保留原始结构指标，同时新增可解释的 bounded/稳定 damage 分数口径。
- 增加 `prettytable` 依赖，用 PrettyTable 打印 group metrics、top risky item 和 top risky prefix。
- 在 `summary.json` / `report.md` 中增加 score normalization 元数据，说明本次 damage 分数的归一化方式。
- 保持现有输出文件名不变，不改变 Hydra analysis runner 入口。

## Capabilities

### New Capabilities

### Modified Capabilities
- `tail-sid-resolution-diagnosis`: 稳定 diagnosis damage score 量级，并改善 stdout/report 表格展示。

## Impact

- Affected code: `src/quantization/tail_sid_diagnosis/metrics.py`, `reporting.py`, tests under `tests/quantization/tail_sid_diagnosis/`.
- Affected dependency files: `pyproject.toml`, `uv.lock`.
- Affected output semantics: `damage`, `tail_damage`, `prefix_risk` 数值量级会变化；原始 collision、near-collision、density 等字段保持不变。
- No changes to train/inference/analysis run-mode dispatch.
