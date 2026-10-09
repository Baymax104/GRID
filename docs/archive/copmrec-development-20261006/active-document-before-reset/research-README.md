# CoPMRec 生成式推荐研究工作区

本项目维护 CoPMRec 论文的研究状态、实验记录、结构化证据、文献和 LaTeX 稿件。实验代码与运行配置由相邻的 `E:\projects\GRID` 仓库维护；本仓库不复制实验实现。

## 当前状态

- 论文范围：利用商品内容改善生成式推荐中的前缀决策。
- 当前方法：CoPMRec，以合法 Semantic ID 子树上的内容概率质量指导训练和 beam decoding。
- 当前阶段：补全论文实验矩阵并持续完善英文稿件。
- 当前证据：Beauty / seed 42 已形成单实例效果和机制证据；跨 seed、跨数据集及部分基线仍待完成或核验。
- 运行边界：新的完整训练、推理和实验矩阵均由用户手动启动。

机器可读状态以 [research-state.yaml](research-state.yaml) 为准，当前实验安排以 [实验矩阵](docs/2026-09-25-copmrec-experiment-matrix.md) 和 [当前计划](ideas/current-plan.md) 为准。历史文档只用于追溯，不自动恢复已经停止的路线或实验预算。

## 推荐阅读顺序

1. [research-state.yaml](research-state.yaml)：研究范围、实验状态、证据和停止决定。
2. [ideas/current-plan.md](ideas/current-plan.md)：当前路线、方法边界与下一步。
3. [docs/2026-09-25-copmrec-experiment-matrix.md](docs/2026-09-25-copmrec-experiment-matrix.md)：论文实验矩阵和完成情况。
4. [docs/dasfaa-2027-execution-board.md](docs/dasfaa-2027-execution-board.md)：论文与实验执行看板。
5. [src/copmrec.tex](src/copmrec.tex)：英文论文主稿。
6. [literature/2026-09-24-copmrec-content-generative-review.md](literature/2026-09-24-copmrec-content-generative-review.md)：相关工作与定位。

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
