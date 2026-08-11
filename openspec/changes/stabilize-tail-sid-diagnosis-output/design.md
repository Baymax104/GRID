## Context

当前 diagnosis 使用 robust z-score：

```text
z(x) = (x - median(x)) / (IQR(x) + eps)
```

当某个指标在大量 item 上近似常数时，`IQR ~= 0`，非中位数样本会被 `eps` 放大，导致 `damage` 和 `prefix_risk` 出现 `1e7+` 量级。这会让用户误以为分数本身代表严重程度，而实际上是归一化失稳。

同时 stdout 表格由字符串拼接生成，列宽不对齐，也没有清楚区分 group summary 与 top risk preview。

## Goals / Non-Goals

**Goals:**
- 让综合分数在常见数据分布下保持可读、稳定、可比较。
- 保留所有原始结构指标，避免只靠综合分数解释实验。
- 使用 PrettyTable 打印整齐的 group metrics 和 top risk preview。
- 在报告和 summary 中记录 normalization 方法，方便解释分数口径。

**Non-Goals:**
- 不重新定义 full collision、near collision、local density 等原始指标。
- 不引入推荐相关性分析。
- 不改变 analysis runner / Hydra 配置入口。
- 不把 PrettyTable 用于 CSV/JSON 文件格式。

## Decisions

### Decision 1: Degenerate IQR falls back to zero contribution

当某个指标的 IQR 小于阈值时，该指标对 z-score damage 的贡献设为 0，而不是除以 `eps`。这样表示“该指标在当前数据集上缺少分辨力”，不应放大为主导风险。

Alternative considered: 使用标准差替代 IQR。标准差仍可能被极端值影响，且 near-constant 情况下也会不稳定。

### Decision 2: Clamp per-metric z contribution

对非退化指标的正向风险贡献做有限范围 clamp，例如 `[0, 5]`。低于中位数的样本不扣分，高于中位数的样本最多贡献 5 分，这能防止单个指标极端值完全吞没其他结构信号，也让 `damage` 更容易解释为非负风险分。

Alternative considered: 完全不 clamp。对诊断排序有时敏感，但不利于用户理解分数量级。

### Decision 3: Keep raw metrics and expose score metadata

`damage` 是综合诊断优先级，不替代原始指标。`summary.json` 和 `report.md` 记录 score normalization 策略，例如 degenerate IQR 阈值、clamp 范围、参与指标。

Alternative considered: 只改计算不输出元数据。这样用户仍难解释分数含义。

### Decision 4: PrettyTable only for stdout

PrettyTable 用于 stdout 展示，不影响机器可读输出。CSV/JSON/Markdown 继续由现有输出逻辑生成。

Alternative considered: 同时用 PrettyTable 生成 Markdown。PrettyTable 更适合终端；Markdown 报告继续保持原生 Markdown 表格。

## Risks / Trade-offs

- [Risk] 分数口径变化会让旧运行结果不可直接比较 -> 在 summary/report 中写明 normalization version。
- [Risk] clamp 可能压缩极端高风险 item 的差异 -> 原始指标仍保留，top item 可结合 raw fields 解释。
- [Risk] 新依赖可能影响环境同步 -> 使用轻量依赖 `prettytable`，并通过 `uv` 更新 lock。
