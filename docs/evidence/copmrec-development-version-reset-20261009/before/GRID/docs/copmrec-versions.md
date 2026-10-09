# CoPMRec 正式版本与历史版本

2026-10-06，用户确定 **v5.3 为正式 CoPMRec**。2026-10-07 明确论文正式核心基线为 **LIGER hybrid**，LIGER dense 仅为内部对照；完整方法的贡献不以相对 v5.2 的单组件增量为判定条件。

## 正式入口

- 训练：仓库根目录 `copmrec_train.sh`，对应 `experiment=copmrec_train`，双卡从头连续 50k。
- 推理：`copmrec_inference.sh --split validation|testing`，对应 `experiment=copmrec_inference`，单卡全目录联合 dense 评分。
- 模型：`src/recommendation/copmrec/module.py` / `configs/model/copmrec.yaml`。数值行为保持 v5.3；正式 checkpoint 新增来源契约，拒绝无契约的开发 checkpoint。
- [正式版本定义](../../research/docs/copmrec-formal-version-20261006.md)、[新正式实验计划](../../research/docs/copmrec-formal-experiment-plan-20261006.md)。

## 正式实验状态

Beauty / Sports / Toys × seed42 / 200 / 2026，CoPMRec 9 个单元与 LIGER hybrid 9 个配对单元均重新执行；目前已启动与完成均为 **0**。开发阶段 run、checkpoint 和指标不能作为新正式完成记录，包括 v5.3 的 `hqw189d2` / `6u6g62hk`。LIGER hybrid 保留 original 生成20候选、与全部cold取并集、content排序；只统一输入历史资格，不换成dense。内部dense仅复用本轮新LIGER checkpoint作单独评价。

上游内容、量化器与 SID 可按冻结来源作为技术依赖复用，执行前记录实际 digest 和输入身份；它们不计为本轮推荐训练、Validation 或 Testing 完成。

## 历史版本

- [历史归档入口](archive/copmrec-development-20261006/README.md)。
- [各版本做法、效果与原矩阵记录](archive/copmrec-development-20261006/versions.md)。
- [2026-10-07 W&B 删除前完整效果登记](archive/copmrec-development-20261006/versions.md#2026-10-07-删除前完整效果登记)：开发 run/Artifact 按用户授权清理，历史效果与身份保存于本地，线上历史链接可能已失效。
- [历史 run 登记表](archive/copmrec-development-20261006/development-run-registry.json)。
- [清理前源码及配置 manifest](archive/copmrec-development-20261006/runtime/manifest.json)。

旧文档中的版本选择、完成状态及停止建议仅说明其发生时状态；当前以本页、正式计划与 research-state.yaml 为准。保留旧源码和实验事实，不复用它们填充新正式矩阵。
