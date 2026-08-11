## Context

`add-tail-sid-diagnosis-core` 已提供 `DiagnosisResult`、CSV/JSON 输出和 stdout 分组摘要。当前增强只关注人类可读报告，帮助研究者快速查看一次运行中最值得分析的 tail damage 证据。

## Goals / Non-Goals

**Goals:**
- 基于内存中的 `DiagnosisResult` 直接生成 `report.md`。
- 报告包含 summary、group metrics、top risky items、top risky prefixes 和输出文件索引。
- CLI 支持 `--top-k-report` 控制 top risk 展示数量。
- stdout 展示 `report.md` 路径和 top risk 预览。

**Non-Goals:**
- 不生成图片或 HTML。
- 不新增 recommendation correlation 分析。
- 不改变 CSV/JSON 输出字段。

## Decisions

### Decision 1: Markdown report in reporting layer

`report.md` 由 `reporting.py` 根据 `DiagnosisResult` 生成，与 CSV/JSON 输出共用同一份结果对象，避免重新加载和二次计算。

Alternative considered: 从落盘 CSV 再读回生成报告。该方式增加 I/O 和解析复杂度，也容易让报告与内存结果不一致。

### Decision 2: Top risk views use existing scores

Top risky items 使用 `tail_damage` 排序，top risky prefixes 使用 `prefix_risk` 排序。这样报告展示与后续 repair/reranking 接口一致。

Alternative considered: 为报告另定义展示分数。该方式会制造第二套 ranking 口径，不利于解释。

## Risks / Trade-offs

- [Risk] Markdown 表格在 top-k 过大时变长 -> 默认 top-k 较小，并允许 CLI 参数调整。
- [Risk] 报告没有图形化分布 -> 保持第二个 change 轻量，后续可独立提案增加 plots。
