> 2026-10-09后续决定：用户选择 **v2为当前正式版本**，删除native view/native CE，训练与推理均无历史排除。原v0/v1与以下阶段记录保留，A3仅作已核验选型依据，不改历史来源；本轮0新增正式运行。最新权威为 [v2正式定义](../../research/docs/copmrec-v2-formal-definition-20261009.md)。

# CoPMRec 版本登记｜v0 / v1（2026-10-09 编号）

> 2026-10-09：v1 Beauty42五臂无排除消融及M1已重新评价并核验，0新训练。见 [v1消融实证](../../research/docs/copmrec-v1-ablation-evaluation-20261009.md)；原v0结果保持，v1完整主矩阵未在本次运行。

CoPMRec 已重新进入开发阶段。原正式实现 **v5.3 登记为 v0**；当前开发版本 **v1** 为训练与推理均不排除历史商品。

| 版本 | 训练排除历史 | raw Validation 排除历史 | 最终推理排除历史 | 当前身份 |
| -- | -- | -- | -- | -- |
| v0 | 否 | 否 | 是 | 原正式参考；主矩阵 9/9 与 M3 已完成 |
| v1 | 否 | 否 | 否 | 开发；Beauty42 无排除推理及匹配 dense 对照已核验 |

v0 训练原本不排除；v1 训练、损失、参数和 own-best 规则相同，仅关闭最终排序排除。历史输入与残差保留。现有训练可按来源审计复用，版本登记不要求新训练。

## 当前文档与执行边界

- [v0/v1 定义](../../research/docs/copmrec-v0-v1-definition-20261009.md)、[开发计划](../../research/docs/copmrec-development-plan-20261009.md)、[机器状态](../../research/research-state.yaml)。
- [v0 原正式定义](../../research/docs/copmrec-formal-version-20261006.md)、[原执行计划](../../research/docs/copmrec-formal-experiment-plan-20261006.md)、[来源登记](../../research/docs/copmrec-formal-run-registry-20261007.md)。
- `copmrec_train.sh` / `experiment=copmrec_train`：原训练入口，双卡；`copmrec_inference.sh`：原正式推理，单卡，有历史排除。
- `copmrec_history_control_inference.sh` / `experiment=copmrec_history_control_inference`：已验证无排除单卡推理入口。原 internal 元数据保持，不宣称默认正式入口已改为 v1。
- 本次仅更新文档和计划；未修改代码、配置或 W&B 元数据，未启动新运行。当前代码/checkpoint 身份仍为 `v5.3`，正式来源契约不重写。

## 已验证来源

原正式 CoPMRec 9 个训练/Testing 与原 LIGER hybrid 9 个配对已核验完成；M3 为 Beauty42 五臂训练/250k、五次 Testing 与机制证据。原 LIGER hybrid 保留原生候选与历史规则，不称为采用共同历史排除。

v0 Beauty42：`gshpyn49` / `vosmuihm`。v1 Beauty42：同一训练 own-best / `hc8oct43`（BMX-148）；共同 dense 无排除 LIGER：`35ig0tz6` / `lu8oct42`（BMX-149）。v1 目前不是完成的多数据集多 seed 矩阵；原 v0 有排除指标不能改写为 v1。

## 旧编号与历史边界

本页 v0/v1 是 **2026-10-09 新编号**；2026-10-06 前旧开发文档中的同名版本按其历史日期解释。Artifact 的 `v0/v1` 与方法版本不是同一概念，不重命名已有 run、URI 或 Artifact。

- [旧开发归档](archive/copmrec-development-20261006/README.md)、[旧版本效果与删除登记](archive/copmrec-development-20261006/versions.md)。
- [历史 run 登记](archive/copmrec-development-20261006/development-run-registry.json)、[归档源码 manifest](archive/copmrec-development-20261006/runtime/manifest.json)。

原正式 v0 和近期匹配对照保留为当前有效证据；早期归档开发路线仍仅供追溯，不随本次进入开发阶段恢复。
