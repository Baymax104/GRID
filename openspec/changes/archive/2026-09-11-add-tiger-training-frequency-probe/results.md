# TIGER 条件分支频次重加权方向验证结果

## 结论

2026-09-12：本提案的工程实现与三组 continuation seed 方向验证已完成。当前第 2–3 层条件分支重加权在三个 seed 中都使 Tail + Cold Hit@10 相对同 seed CE 对照提高 0.0319 个百分点，即 Tail 命中数均由 35 增至 36；但每组仅净增 1 个 Tail 命中，Overall 排序方向不稳定，预声明的第 2 层机制门槛仅在 seed 42、2024 通过。当前版本停止扩展，不进入跨数据集或参数 sweep。

归档表示探针、审计协议和实验结论完成，不表示训练侧主方法成立。后续若继续训练侧研究，应先检查条件分支权重是否真正对准 Tail；不得把本结果表述为频率偏置的因果验证。

## 实验身份

所有训练均从 Beauty / RKMeans checkpoint `wandb://26qh50do` 和 Semantic ID `wandb://4vyi4o6w` 初始化，使用 fresh Adam、lr=5e-5、2,000 steps、两卡 global batch 256、layers=[2,3]、alpha=0.25、cap=2.0。两个 arm 共享 175 个 training 文件摘要、SID 指纹和权重 lookup。训练后的评估统一使用 Beauty evaluation、beam=10、evaluation seed 42；testing 未使用。

| Continuation seed | Arm | Train run | Trace run | Diagnosis run | Evidence Artifact |
|---:|---|---|---|---|---|
| 42 | CE | `1gzt5yvd` | `rqu8kvu4` | `ggxh0nrz` | `tail-sid-diagnosis-evidence:v22` |
| 42 | Reweighted | `6rc9rbos` | `pzxr7416` | `7pvhlis6` | `tail-sid-diagnosis-evidence:v23` |
| 2024 | CE | `lu87lzit` | `ruoouv5b` | `eq4lth7f` | `tail-sid-diagnosis-evidence:v24` |
| 2024 | Reweighted | `13rjr32d` | `gwn4r365` | `jnh6tuy1` | `tail-sid-diagnosis-evidence:v26` |
| 2025 | CE | `9lijvxwk` | `u52de35e` | `imjlyz5i` | `tail-sid-diagnosis-evidence:v25` |
| 2025 | Reweighted | `js8b9t82` | `b35vs0ht` | `m3f0e9be` | `tail-sid-diagnosis-evidence:v27` |

六份 recommendation evidence 的 22,363 个 user keys、目标 item、SID 和分组完全对齐。W&B summary 的 `trainer/global_step=1999` 是日志步号；六个训练 checkpoint 的内部 `global_step` 均为 2,000。

## 最终推荐结果

以下差值均为 Reweighted − 同 seed CE，百分点记为 pp。

| Seed | Overall Hit@10 Δ | Head Hit@10 Δ | Mid Hit@10 Δ | Tail + Cold Hit@10 Δ | Overall 净命中 | Head 净命中 | Mid 净命中 | Tail 净命中 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.0447 pp | -0.0628 pp | +0.1981 pp | +0.0319 pp | +10 | -7 | +16 | +1 |
| 2024 | -0.0537 pp | -0.2153 pp | +0.1362 pp | +0.0319 pp | -12 | -24 | +11 | +1 |
| 2025 | -0.0179 pp | -0.1166 pp | +0.0990 pp | +0.0319 pp | -4 | -13 | +8 | +1 |
| 三组平均 | -0.0089 pp | -0.1316 pp | +0.1444 pp | +0.0319 pp | -2.0 | -14.7 | +11.7 | +1.0 |

三组 Tail + Cold NDCG@10 均提高，Overall/Head 损失均在预声明的 0.2/0.5 pp 门槛内。不过，稳定且较大的收益集中在 Mid。Tail 的状态迁移分别为新增/丢失 2/1、1/0、1/0；重加权后每组只命中 7–9 个不同 Tail item，覆盖仍窄。三个 seed 没有共同新增的同一用户，seed 42 与 2025 共享一个新增目标。

与原 checkpoint 的同口径 anchor 相比，CE 继续训练已经贡献主要收益：三个 CE run 的 Tail 命中均从 29 增至 35；重加权只在此基础上增至 36。不能把 warm-start 的整体提升归因于重加权。

## 层级机制结果

预声明机制门槛要求 Tail 第 2 层平均合法排名降低且合法 Top10 比例提高。

| Seed | 第 2 层合法排名 CE → Reweighted | 第 2 层合法 Top10 CE → Reweighted | 判定 |
|---:|---:|---:|---|
| 42 | 17.8944 → 17.8856 | 39.5334% → 39.7926% | 通过，幅度小 |
| 2024 | 17.9533 → 17.9358 | 39.1769% → 39.4686% | 通过，幅度小 |
| 2025 | 17.8811 → 17.8694 | 39.7602% → 39.6954% | 未通过 |

第 3 层合法 Top10 在 seed 42、2024 略降，seed 2025 持平。三个 seed 的 Tail 第 2、3 层平均目标概率均下降。因此，结果只支持少量边界目标的相对排序发生变化，不支持“模型普遍提高 Tail 分支置信度”或稳定的层级机制。各 diagnosis 的既有 `prefix_survival_mechanism` verdict 仍为 `not_supported`。

## 预声明决策

- 有效性：通过。训练预算、初始化、统计、SID、评估 split 与 user keys 均对齐。
- 推荐效用：通过。三个 seed 的 Tail + Cold Hit/NDCG 均改善，Overall/Head 代价未越界。
- 机制重复性：未通过。第 2 层 Top10 改善只在 2/3 seeds 出现，目标概率方向与预期相反。
- 最终决策：`stop_current_variant`。不新增训练 seed、不调 alpha/cap、不扩展 Sports/Toys；保留独立模块作为可复用探针。

这里的 seed 是同一预训练 checkpoint 上的 continuation seed，不是三个独立 baseline 训练 seed。结果来自单一 Beauty/RKMeans 设置，不提供跨数据集、跨 SID 或独立预训练稳定性证据。

## 工程与验证记录

实现新增独立训练类、期望监督频次统计、薄配置和根脚本；baseline Tiger、dataset、collate、decoder 和统一入口没有原地修改。配置继承故障已通过 `tiger_train@_here_` 修复，并增加真实构造回归。

归档前重新执行完整 pytest（467 passed、4 个上游弃用 warning）、Ruff、`git diff --check` 和 OpenSpec strict。完整训练由用户在服务器手动执行，本地不重复启动 GPU 实验。
