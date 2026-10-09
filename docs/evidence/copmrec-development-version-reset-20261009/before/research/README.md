# CoPMRec 生成式推荐研究工作区

本项目维护 CoPMRec 论文的研究状态、实验记录、结构化证据、文献和 LaTeX 稿件。实验代码与运行配置由相邻的 `E:\projects\GRID` 仓库维护；本仓库不复制实验实现。

## 当前状态

- 论文范围：利用商品内容改善生成式推荐中的前缀决策。
- 当前正式方法：CoPMRec v5.3；保留内容条件前缀混合训练，采用共享协同残差、完整目录 CE 和辅助内容视图 CE，正式部署为全目录 dense 评分。
- 正式基线：LIGER hybrid；LIGER dense 仅是同一新基线 checkpoint 的内部对照，不另训练。正式矩阵为 Beauty / Sports / Toys × seed42 / 200 / 2026。
- 当前阶段：正式版本和重新执行计划已准备；CoPMRec 与 LIGER hybrid 共9个配对、18个新模型单元，已启动、已完成均为0；LIGER dense另列9个内部Testing，不增加训练或Val。
- 历史边界：所有开发 run，包括 v5.3 开发训练与 Validation，只用于历史追溯，不作为本轮已完成实验或正式主表结果。
- 运行边界：新的完整训练、推理和实验矩阵均由用户手动启动。

机器可读状态以 [research-state.yaml](research-state.yaml) 为准，当前实验安排以 [v5.3 正式实验计划](docs/copmrec-formal-experiment-plan-20261006.md) 和 [当前计划](ideas/current-plan.md) 为准。历史文档只用于追溯，旧版完成记录和旧阶段预算不转移到新正式矩阵。

## 推荐阅读顺序

1. [research-state.yaml](research-state.yaml)：研究范围、实验状态、证据和停止决定。
2. [ideas/current-plan.md](ideas/current-plan.md)：当前路线、方法边界与下一步。
3. [正式版本](docs/copmrec-formal-version-20261006.md) 与 [正式实验计划](docs/copmrec-formal-experiment-plan-20261006.md)：固定方法、配对矩阵和新完成门禁。
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
