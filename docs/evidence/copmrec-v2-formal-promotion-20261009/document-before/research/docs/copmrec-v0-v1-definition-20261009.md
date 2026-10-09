# CoPMRec v0 / v1 版本定义（2026-10-09）

> v1现有证据已补充Beauty42五臂无排除Testing及M1重新分析，原五臂训练/own-best复用；见 [v1消融实证](copmrec-v1-ablation-evaluation-20261009.md)。仍不是三数据集多seed无排除矩阵。

用户于 2026-10-09 将 CoPMRec 重新置于开发阶段：原正式版本登记为 **v0**，训练与推理均不排除历史商品的版本登记为 **v1**。当前开发目标为 v1，v0 保留为原正式参考版本。

## 版本与实际计算

| 项目 | v0：原正式参考 | v1：当前开发 |
| -- | -- | -- |
| 对应实现 | 原 v5.3 | 沿用原 v5.3 训练与评分，关闭最终历史排除 |
| 训练是否排除历史商品 | 否 | 否 |
| 训练内 raw dense Validation 是否排除 | 否 | 否 |
| Testing / 推荐最终排名是否排除 | 是 | 否 |
| 历史商品作为输入、内容及残差表示 | 保留 | 保留 |
| 损失、参数和训练预算 | 原正式协议 | 与 v0 相同 |
| checkpoint 选择 | 训练内 raw full-catalog val NDCG@10 首次最优 | 同一规则，不按 Testing 重新选点 |
| 当前证据覆盖 | 三数据集 × 三 seed 主矩阵和 Beauty42 M3 | 已核验 Beauty42 无排除推理及匹配 LIGER dense 对照 |

“不排除”指目录损失和最终排序中不屏蔽已交互商品；不删除输入历史、序列建模或历史侧残差。v0 的训练原本就不排除，不存在将“训练排除”关闭这一新训练改动。因此 v1 可复用经来源审计的 v0 own-best checkpoint；版本登记本身不产生新训练。

## 来源与命名

- 这是 **2026-10-09 新版本编号**，与 2026-10-06 前归档开发路线中的旧 v0/v1 无关；旧文档的编号按其日期解释。
- 原 checkpoint / Hydra 运行时身份仍为 `v5.3`，`formal_release_id=copmrec-v5.3`。本轮仅更新文档和计划，不修改来源契约、模型、配置、脚本或已有 W&B metadata。
- Artifact 的 `:v0` / `:v1`、URI 的 `alias=v0` / `alias=v1` 是 Artifact 版本，不是方法版本。
- `copmrec_train.sh` 与 `copmrec_inference.sh` 仍是原正式入口；不能把后者的默认排除行为称为 v1 默认行为。已有 `copmrec_history_control_inference.sh` / `experiment=copmrec_history_control_inference` 实现无排除推理；其既有运行按真实 internal 身份保留。
- 当前登记将 `hc8oct43` 对应的真实计算条件映射为 v1 初始开发证据，保留它原先作为内部对照产生的身份，不追改为新训练或独立确认。

## 固定的初始来源

| 条件 | Training run | Testing run | own-best |
| -- | -- | -- | -- |
| v0 / Beauty42 | `gshpyn49` | `vosmuihm` | step 47500 |
| v1 / Beauty42 | 同一 `gshpyn49` | `hc8oct43` / BMX-148 | 同一 step 47500 |
| LIGER dense / 无排除 | `35ig0tz6` | `lu8oct42` / BMX-149 | step 45000 |
| LIGER dense / 有排除（历史规则对照） | 同一 `35ig0tz6` | `ldi54f1o` / BMX-147 | 同一 step 45000 |

CoPMRec checkpoint：`wandb://baymaxam/GRID/gshpyn49?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=047500.ckpt`，SHA256 `a64ca93d49b5bb9ead7b4af803091346ca62af5299cd89266ac5a2acce3894c4`。

LIGER checkpoint：`wandb://baymaxam/GRID/35ig0tz6?role=checkpoint&alias=v1&file=checkpoint_epoch=000_step=045000.ckpt`，SHA256 `8508e08e2a2cc9ea5d2bbc902aa4e2b6415c8a45b9c8a0ddc879728a6b66b43b`。

两次无排除推理的源码 SHA256：`a034b49abea85191f19cebc9ebafcca9a1426a7db4be81ad961a791c22e4ed2c`。精确输入、输出、指标和原运行命令保留于 BMX-148/149 与各自证据目录。

v0 已完成证据继续有效并可用于当前比较；归档旧开发路线仍仅作追溯，不因本次进入开发阶段而恢复。当前安排见 [开发计划](copmrec-development-plan-20261009.md)。
