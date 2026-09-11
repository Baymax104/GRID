# 探针实现归档记录

## 实现完成与实验结论

2026-09-11：本提案的探针、数据契约、配对分析、启动脚本及回归修复已完成。归档表示工程交付完成，不表示研究方法通过效用门槛。

Beauty / RKMeans / seed 42 / evaluation / beam 10 的有效配对为 baseline `hho25h7i` 与 intervention `23lqbr7z`，diagnosis 为 `sn1lsx3d`，W&B 项目为 `baymaxam/GRID`，证据 Artifact 为 `tail-sid-diagnosis-evidence:v21`。原 intervention `oznapqx8` 因 catalog 外候选被排除。

Tail + Cold Hit@10 提升 0.4463 个百分点，但 Overall 下降 0.3488 个百分点、Head 下降 0.8342 个百分点，超过预声明的 0.2 / 0.5 个百分点损失门槛。当前版本开发设置 no-go；跨设置聚合只有 1/4，仍为 inconclusive，不能宣称完成四设置验证。由于 advance 要求全部通过，不继续以推进该版本为目的扩展剩余设置。

## 已修复问题与证据限制

- 同宽 allocation 配对不再进入要求 widened width 更大的 recovery 分析。
- shortlist padding 不再进入 reserve / score backfill，保留候选必须属于 catalog。
- `target_allocation_shortlisted` 按排序后的候选前缀值判断，不再混淆位置与 token ID。`sn1lsx3d` 的该字段仍是修复前数据，不用于 shortlist-rate 结论；推荐输出、主指标、reserve 迁移和 teacher-forcing 证据不受该 trace 字段缺陷影响。

## 归档验证

提交前 `uv run pytest -q`：431 passed；`uv run ruff check src tests` 通过；归档前 OpenSpec 全量 strict 校验 78 项通过。配置 compose、脚本语法与参数回归包含在测试中。完整实验仍由用户手动启动。

研究解释与方向修订维护于独立研究仓库的 `docs/beauty-rkmeans-negative-result-2026-09-11.md`；本提案不自动启动训练侧方法。
