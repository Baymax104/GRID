# LETTER Beauty/Sports 训练核查与 Testing

## 训练完成核查

Beauty/Sports × seed42/200/2026 六组修复后推荐训练均 `finished`、退出码 0；日志确认达到 50,000 optimizer steps。W&B 最后记录的 `trainer/global_step=49999` 为步数记录口径，不是少训练一步。

每组完整验证 100 次，间隔 500 步；验证集选出的最大 NDCG@10 与 checkpoint artifact 的 best score 一致。逐组核对 best 对应历史步数、文件 SHA256 和 artifact digest，Testing 固定以下版本。

| Issue | Dataset | Seed | Training run | Best step | Best val NDCG@10 | Checkpoint version |
| --- | --- | ---: | --- | ---: | ---: | --- |
| BMX-60 | Beauty | 42 | diiyhjux | 40,500 | 0.047045058 | letter_train_beauty-checkpoint:v0 |
| BMX-61 | Beauty | 200 | qzcesmm6 | 36,500 | 0.045484724 | letter_train_beauty-checkpoint:v2 |
| BMX-62 | Beauty | 2026 | yfdhrh8g | 38,000 | 0.046552254 | letter_train_beauty-checkpoint:v1 |
| BMX-63 | Sports | 42 | 0gokpcne | 41,500 | 0.024214576 | letter_train_sports-checkpoint:v2 |
| BMX-64 | Sports | 200 | wdmgy3lu | 39,500 | 0.023850058 | letter_train_sports-checkpoint:v1 |
| BMX-65 | Sports | 2026 | ktf4xvx1 | 43,500 | 0.024477837 | letter_train_sports-checkpoint:v0 |

训练源码 snapshot 的本地来源标记为 verified，发布文件 digest 与运行端 snapshot 一致。Testing 前再次核对 LETTER 推荐模块和数据适配代码与训练源码 archive 一致；沿用各组相同 SID，不重新选择或修改 checkpoint。

## Testing 执行与修复

通过根目录 `letter_inference.sh`，双卡 `uv run torchrun ... -m src.main experiment=letter_inference` 执行 Testing。seed42 使用物理 GPU0/1，seed200 使用 GPU2/3，seed2026 使用 GPU4/5；各进程组内为 CUDA0/1。同一 GPU pair 同时承载 Beauty 和 Sports，未停止其他任务。

协议为完整 testing split、history20、temperature1、beam20/top10、FP32、每卡 batch32、全部有效用户保留，不过滤历史商品。W&B resolved config 中 checkpoint 已解析为缓存文件，核验时将该文件 SHA256 与训练 best 文件身份对齐，并核对消费的 checkpoint artifact 固定版本。

首轮五组在预测完成后暴露公共 summary writer 的 DDP 错误：非零 rank 的 W&B logger 占位对象返回 `summary` 方法，写入时报 TypeError；另一个 Beauty seed2026 在读取 SID 时遇到 W&B AuthenticationError。首轮六组均退出码 1，其 summary 不作为有效结果。

最小修复位于 `src/common/metrics/callback.py`：所有 rank 继续参与指标 compute/聚合，只有 `trainer.is_global_zero` 写 summary。新增回归测试检查非零 rank 仍计算、重置指标且不访问 logger；12 项聚焦测试和 Ruff 通过。Mutagen flush 后三个 session Watching for changes、无 conflict，随后六组使用新 run 重跑。

## 证据位置

- 训练完成核验：本地及 node1 的 `logs/letter-formal-20261009-prefixfix-aad918p3/completion-audit.json`。
- 首轮失败记录：node1 的 `logs/letter-testing-20261010/`。目录和首轮 output id 的日期后缀是启动标识，实际运行日期为 2026-10-09。
- 修复后 Testing：本地及 node1 的 `logs/letter-testing-20261009-summaryfix/testing-plan.json`；node1 同目录保留每组脚本、日志、退出码。
- 独立核验：同目录 `testing-verification.json`，包含逐组 W&B 状态、退出码、输入 lineage、预测完整性、源码记录和复算指标。
- SID 来源核验：同目录 `sid-lineage-verification.json`；六组消费的 SID artifact 均与训练一致，重新加载后目录 SHA256 与训练记录一致，已全部通过。

Testing 完成后以完整预测 bundle 对照原始 TFRecord 的用户 key 与末尾 target 复算 first-match Recall/NDCG@5/@10；核对用户集合、候选目录合法性、逐用户去重、W&B summary 及发布 output artifact digest。

## 最终 Testing 结果

六组修复后的 Testing 均 `finished`、退出码 0，独立复算与 W&B summary 的绝对差小于 `1e-12`。Beauty 每组完整覆盖 22,363 名用户，Sports 每组完整覆盖 35,598 名用户；预测分别为 `[22363,10]` 和 `[35598,10]`，全部商品在对应目录内，逐用户无重复候选。

| Dataset | Seed | Testing run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Beauty | 42 | [wvpeeti8](https://wandb.ai/baymaxam/GRID/runs/wvpeeti8) | 0.039305997 | 0.025693202 | 0.060009838 | 0.032364410 |
| Beauty | 200 | [da4bbei4](https://wandb.ai/baymaxam/GRID/runs/da4bbei4) | 0.040736932 | 0.027097455 | 0.063453025 | 0.034372072 |
| Beauty | 2026 | [jhdcssd1](https://wandb.ai/baymaxam/GRID/runs/jhdcssd1) | 0.045298037 | 0.029820381 | 0.069847516 | 0.037718625 |
| Sports | 42 | [3iia84nm](https://wandb.ai/baymaxam/GRID/runs/3iia84nm) | 0.021967526 | 0.014012800 | 0.034608686 | 0.018074308 |
| Sports | 200 | [efy52sdx](https://wandb.ai/baymaxam/GRID/runs/efy52sdx) | 0.020422496 | 0.013429234 | 0.031939997 | 0.017143913 |
| Sports | 2026 | [74u0edre](https://wandb.ai/baymaxam/GRID/runs/74u0edre) | 0.021349514 | 0.013513263 | 0.032670375 | 0.017154379 |

三 seed 均值 ± 样本标准差（ddof=1）：

| Dataset | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| Beauty | 0.034818369 ± 0.002704864 | 0.064436793 ± 0.004992077 |
| Sports | 0.017457534 ± 0.000534168 | 0.033073019 ± 0.001379155 |

全部最终 recommendation output artifact 为 COMMITTED。Sports seed2026 的本地 writer 与 artifact writer 文件字节不同，按用户 key 对齐后用户与全部候选完全一致；发布文件 digest 与 artifact manifest 一致。因此以 bundle 内容身份核对两种 writer，以各自实际文件核对发布 digest。

修复后 Testing 源码 SHA256 为 `45635c00cba54eb43df3582c163ebde52dce53130198d98f75a6df2cb552c2b9`，各组 source artifact 来源 verified、发布文件 digest 核验通过。当前结果为固定 GRID 协议下的正式 Testing 指标；论文使用时仍需遵守所有 baseline 的共同比较协议。
