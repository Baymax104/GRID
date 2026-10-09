# 代码库整理与变更归档（2026-10-09）

## 归档结果

本次归档 68 个任务清单已完成的 OpenSpec 变更，目录为 `openspec/changes/archive/2026-10-09-<change-id>/`。其中 22 个当前能力合并到正式 specs；46 个历史或被取代的变更使用 `--skip-specs` 保留原文，不恢复已退役要求。原 proposal、design、tasks 和 delta 保持归档时内容。

`validate-cgbs-content-routing-mechanism` 的任务 4.2 尚未完成，继续保留在 changes 中；不补勾、不标记完整机制资格成立。`add-letter-cf-test` 只有空目录，没有文件或任务清单，Git 不跟踪。

### 合并的当前能力

- `add-liger-candidate-trace`
- `add-liger-content-model`
- `add-liger-hybrid-retrieval`
- `integrate-liger-pipeline`
- `add-mutagen-code-sync`
- `add-sasrec-backbone`
- `add-sasrec-data`
- `add-sasrec-training-evaluation`
- `add-sasrec-pipeline`
- `add-run-source-snapshot`
- `optimize-tiger-prefix-membership`
- `add-letter-recommender`
- `add-letter-data`
- `add-letter-training-evaluation`
- `add-letter-pipeline`
- `add-letter-tokenizer`
- `add-letter-cf-teacher`
- `fix-letter-sid-collision-export`
- `promote-copmrec-joint-only-formal`
- `consolidate-copmrec-v2-formal-surface`
- `optimize-letter-prefix-constraints`
- `optimize-letter-tokenizer-diversity-sampling`

### 仅保留历史记录

- `add-tiger-catalog-grounded-scoring`
- `fix-item-resolution-validation-performance`
- `add-tiger-item-resolution`
- `add-item-resolution-score-search-audit`
- `add-bounded-residual-item-retrieval`
- `add-brir-training-diagnostics`
- `validate-sid-initialization-value`
- `explore-sid-partition-learnability`
- `validate-sid-partition-intervention`
- `diagnose-cgbs-content-learning`
- `diagnose-cgbs-candidate-score-cross`
- `train-cgbs-content-warmup`
- `train-cgbs-frozen-a-probe`
- `diagnose-cgbs-frontier-search-transfer`
- `qualify-cgbs-root-content-information`
- `diagnose-cgbs-item-root-localization`
- `expand-fixed-content-capacity`
- `add-liger-content-guided-candidates`
- `add-liger-probability-mixture`
- `add-liger-paired-rerank`
- `add-liger-learned-reranker`
- `add-liger-dynamic-mixture`
- `control-liger-prefix-mechanism`
- `complete-liger-training-decoding-factorial`
- `diagnose-liger-preference-dispersion`
- `validate-liger-depth-conditioned-aggregation`
- `validate-liger-candidate-union`
- `diagnose-liger-source-protection`
- `validate-liger-single-max-slot`
- `qualify-cgbs-content-representation`
- `qualify-cgbs-first-failure-objective`
- `train-cgbs-branch-residual-probe`
- `train-liger-joint-probability-mixture`
- `add-cgbs-attention-warmup`
- `freeze-cgbs-content-after-warmup`
- `validate-a-role-sharing`
- `isolate-cgbs-auxiliary-encoder-gradient`
- `add-cgbs-state-dependent-gate`
- `add-a-score-search-audit`
- `add-second-level-content-interaction`
- `add-liger-path-trace`
- `add-liger-budget-matched-continuation`
- `promote-copmrec-v53-formal-version`
- `add-copmrec-m3-experiment-entrypoints`
- `add-copmrec-history-inference-control`
- `evaluate-copmrec-v1-ablations`

## 当前规格校正

- 当前方法范围包括 TIGER、LIGER、CoPMRec、LETTER 和 SASRec。
- CoPMRec 使用正式 v2 三目标、全阶段无历史排除及 no_mixture/no_residual/no_joint_ce 三臂；旧 JointMixtureLiger 和开发版本恢复要求不再适用。
- 实验状态按实际核验登记，不将迁移时的 Todo 状态写成永久要求；迁移不授权实验、不重置预算。
- 路径观察规格只描述当前可达的 LIGER probability_mixture 与 CoPMRec learned_mass，不恢复旧 legal/max 控制。
- Mutagen 仍使用三个窄端点，根目录白名单明确包含 PowerShell 脚本。

## 提交分组

按依赖顺序分别提交仓库规则、共享 Artifact 与指标设施、源码快照、TIGER 前缀性能、TIGER 训练推理协议、SASRec、LETTER、CoPMRec/LIGER 正式运行面、历史探索归档，以及说明文档与索引。每组附带其相关测试、配置和当前规格。

## 验证与保存边界

各模块已执行聚焦 CPU pytest、相关 Ruff、Hydra compose、脚本参数及 shell 语法检查；源码快照和同步脚本通过 PowerShell Parser 检查，uv.lock 通过离线一致性检查。归档后的 OpenSpec strict 全量校验为 107/107 通过（106 个 specs、1 个活动 change）；完整 pytest 回归为 825 passed；仅有既有第三方弃用提示。功能交付归档不等于正式训练或 Testing 完成。

按用户选择，Git 仅保存 Markdown 说明与归档索引，原始运行证据保留本地；退役入口回归所需的 21,746 B 源码 ZIP 与 manifest 是固定测试输入，作为唯一例外随代码保存。没有删除本地证据，没有提交或修改邻近 research 仓库，没有远端同步或实验操作。

历史 Markdown 原文保留既有末尾空行和换行空格；这些文档空白不参与功能验证。源码、配置和当前规格通过提交前的 whitespace 检查。

## docs 版本控制复核

第一轮规则将全部 Markdown 递归纳入 Git，带入了文档副本、运行目录中的规格快照、迁移前计划和临时 issue 内容。复核后从当前 Git 索引移除 194 个文件，本地原件及其字节保持不变；docs 的跟踪文件从 299 个缩减为 105 个。

保留根目录说明、归档目录的直接总结与索引、证据 README，以及正文实际引用的八份指标、诊断和命令说明。`.gitignore` 改用上述有限范围，防止后续再次把所有 Markdown 快照纳入版本控制。

同时取消此前退役源码 ZIP/manifest 的例外，退役入口回归改为不依赖本地归档文件。历史 ZIP 完整性核验仍可使用本地原件执行；代码单元测试仅验证当前运行面。

本次相关 pytest 为 45 passed，Ruff 和 staged whitespace 检查通过。另从暂存树导出仅含 Git 跟踪文件的检出，在没有退役 ZIP/manifest 的情况下重跑相同测试，仍为 45 passed。移除跟踪的 194 个本地文件逐一核对 SHA256，均未改变；非冻结归档说明的 Markdown 引用未新增断链。
