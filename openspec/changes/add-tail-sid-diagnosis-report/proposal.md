## Why

核心诊断已经能输出机器可读 CSV/JSON，但研究迭代需要一个可直接阅读的运行报告来快速判断 tail damage 是否存在、哪些 item 和 prefix 最危险。该增强专注于结果展示，不改变指标计算契约。

## What Changes

- 为 Tail-SID 诊断输出新增 `report.md`，汇总关键指标、分组对比、top risky items 与 top risky prefixes。
- CLI 结束时展示 `report.md` 路径和 top risk 预览。
- 增加 `--top-k-report` 参数控制报告中展示的 item/prefix 数量。
- 不改变已有 CSV/JSON 文件名和字段。

## Capabilities

### New Capabilities

### Modified Capabilities
- `tail-sid-resolution-diagnosis`: 增加人类可读 Markdown 报告和 top risk 展示要求。

## Impact

- Affected code: `src/quantization/tail_sid_diagnosis/reporting.py`, `src/quantization/tail_sid_diagnosis/run.py`, tests under `tests/quantization/tail_sid_diagnosis/`.
- APIs: 新增可选 CLI 参数 `--top-k-report`。
- Dependencies: 不新增依赖。
