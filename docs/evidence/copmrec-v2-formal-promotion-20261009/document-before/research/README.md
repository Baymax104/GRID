# CoPMRec 生成式推荐研究工作区

> 2026-10-09 v1消融追加评价已核验完成：五臂复用原own-best，5次单卡Testing与一次M1零forward分析，0新训练。见 [v1消融结果](docs/copmrec-v1-ablation-evaluation-20261009.md)。下文初始/文档阶段边界按其日期解释，未新增多数据集多seed矩阵。

本项目维护 CoPMRec 论文的研究状态、实验记录、结构化证据、文献和 LaTeX 稿件。实验代码与运行配置由相邻的 `E:\projects\GRID` 仓库维护；本仓库不复制实验实现。

## 当前状态

- 当前阶段：2026-10-09 按用户决定进入 v1 开发，当前正式参考版本登记为 v0（原实现 v5.3）。
- v0 训练不排除、推理排除；v1 训练和推理均不排除。两者 raw Validation、训练损失与 own-best 规则相同。
- 已完成证据：v0 三数据集 × 三 seed 主矩阵 9/9、原 LIGER hybrid 9 配对，以及 Beauty42 M3 五臂/250k 与机制包保留。
- v1 初始证据：Beauty42 `hc8oct43`；共同无排除 LIGER dense 对照 `lu8oct42`，均复用各自原 own-best，无新训练。尚无 v1 三数据集三 seed 主矩阵。
- 本轮范围：更新文档、研究状态与 Linear 计划，0 新训练/推理/diagnosis。后续完整运行仍由用户明确启动。
- 来源边界：既有 W&B、checkpoint 和配置中的 v5.3 身份保持；早期归档开发路线不自动恢复。

机器可读事实以 [research-state.yaml](research-state.yaml) 为准，当前安排以 [v1 开发计划](docs/copmrec-development-plan-20261009.md)、[v0/v1 定义](docs/copmrec-v0-v1-definition-20261009.md) 和 [当前计划](ideas/current-plan.md) 为准。

## 推荐阅读顺序

1. [research-state.yaml](research-state.yaml)：版本、阶段、有效证据和执行边界。
2. [开发计划](docs/copmrec-development-plan-20261009.md) 与 [版本定义](docs/copmrec-v0-v1-definition-20261009.md)。
3. [当前计划](ideas/current-plan.md) 与 [执行看板](docs/dasfaa-2027-execution-board.md)。
4. [来源登记](docs/copmrec-formal-run-registry-20261007.md) 与 [共同无排除对照](docs/copmrec-liger-both-history-off-20261008.md)。
5. [src/copmrec.tex](src/copmrec.tex)：英文论文主稿；本文档变更未修改稿件。
6. [相关工作与定位](literature/2026-09-24-copmrec-content-generative-review.md)。

## 目录职责

| 路径 | 内容 |
|---|---|
| `src/` | LaTeX 主稿、参考文献、表格和表格生成脚本 |
| `docs/grid-experiments/` | 从 GRID 迁移的实验协议、结果、图表和结构化证据 |
| `docs/evidence/` | 早期研究阶段保留的结构化证据 |
| `docs/` | 当前研究判断、项目规则和历史实验报告 |
| `ideas/` | 当前计划、方法设计与历史候选路线 |
| `literature/` | 文献笔记、论文 PDF 与毕业和投稿要求 |
| `baseline/` | 核心 baseline 原始材料 |

## 论文构建

从 `src/` 目录运行：

```powershell
latexmk -pdf -outdir=build copmrec.tex
```

生成文件统一写入 `src/build/`，该目录不纳入 Git。正文中的实验数字应能追溯到 `research-state.yaml`、W&B run/Artifact 或 `docs/grid-experiments/evidence/` 中的结构化记录。

## 维护规则

- Markdown 文档以中文为主，论文正文使用英文。
- 当前结论变化时同步更新研究状态、当前计划和稿件。
- 不将 evaluation/testing 上的事后选择写成独立确认。
- 不提交临时文件、LaTeX 构建产物、虚拟环境或凭据。
- 研究协作和证据门禁详见 [AGENTS.md](AGENTS.md)。
