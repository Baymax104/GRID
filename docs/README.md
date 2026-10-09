# GRID 文档索引

## 当前入口

- [CoPMRec 正式版本](copmrec-versions.md)：当前 v2 的模型、三项训练目标、三臂消融和机制分析入口。
- [LETTER 实现](letter.md)与[验证记录](letter-validation.md)：CF teacher、Tokenizer、SID 导出及推荐链路。
- [SASRec 基线](sasrec-baseline.md)与[实现计划](sasrec-implementation-plan.md)：数据、算法骨干和训练推理协议。
- [运行源码快照](run-source-snapshot.md)：本地可信来源、运行端字节归档与 W&B code Artifact。
- [2026-10-09 整理记录](repository-cleanup-20261009.md)：OpenSpec 归档范围、规格合并边界和提交分组。

当前研究版本、实验状态和预算以 `../../research/research-state.yaml` 与 `../../research/ideas/current-plan.md` 为准。代码行为以当前 `src/`、`configs/` 和根目录脚本为准。

## 历史记录

带日期的实验文档、`archive/` 和 `evidence/` 中的说明保留记录时的事实，旧文档中的“当前版本”、命令和任务状态不代表今日运行面。尤其是早期 `copmrec-v2.md` 所指的开发版本，不等同于 2026-10-09 的正式 v2；请从上面的正式版本索引进入。

归档保存历史事实，不恢复已退役实现、旧研究结论或预算。功能变更归档也不表示对应正式训练和 Testing 已完成。

## 文件保存范围

Git 保存 Markdown 说明与归档索引。原始 JSON、CSV、日志、图片、论文附件及历史源码快照保留在本地，说明中的这些相对链接需要相应本地文件才能打开。

唯一例外是 `archive/copmrec-conversion-entrypoints-20261003/manifest.json` 和 `retired-files.zip`：它们合计约 22 KB，是退役入口完整性测试的固定输入，随代码版本保存。此规则不删除任何本地证据，也不执行远端同步或实验。
