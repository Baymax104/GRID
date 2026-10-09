# SASRec 分模块实现计划

## 目标与边界

在 GRID 统一入口实现用于正式基线比较的 SASRec；算法依据作者 `kang205/SASRec@e3738967fddab206d6eeb4fda433e7a7034dd8b1`。用户授权自主分模块提案与实施，以及本地 dry-run/5 step 逻辑验证；完整实验仍手动启动。

## 模块顺序

| 模块提案 | 职责 | 验收 | 状态 |
|---|---|---|---|
| add-sasrec-backbone | 官方 attention/FFN、embedding、BCE/L2 | 独立公式对照、因果性、梯度 | 完成，12项测试及 Ruff/strict 通过 |
| add-sasrec-data | 固定目录映射、左 padding、移位标签、负采样、batch | ID 0、稀疏 ID、截断、负样本边界 | 完成，数据/算法24项测试及 Ruff/strict 通过 |
| add-sasrec-training-evaluation | Lightning、optimizer、全目录分块排名、metrics、checkpoint 身份 | 训练梯度、全库排名一致、选模/产物契约 | 完成，累计36项测试及 Ruff/strict 通过 |
| add-sasrec-pipeline | Hydra configs、脚本、说明、dry-run/5 step | compose、参数/语法、实际统一入口运行 | 完成，累计54项 SASRec 测试与17项共享回归通过；本地 CUDA 验证完成 |

每个提案只改变对应模块；依赖已有模块，不把完整实现合并成一个提案。现有用户修改保持。

## 已固定的算法

Q 来自 LN(x)，K/V 来自 x，attention 无 output projection，residual 加 LN(x)；FFN 为 hidden→hidden→hidden，residual 加其归一化输入。位置按固定槽位，左 padding，最终槽位表示用于点积打分。正负 BCE 监督每个有效位置，保留官方 1e-24，L2 同时覆盖 raw item 和 position embedding，optimizer 使用官方 Adam/beta2=0.98。

## 后续协议决策

数据沿用 training/evaluation/testing；评价目标是最后商品、只输入前序历史。目录与生成式方法一致，ID 0 是真实商品，在模型内重映射为非零。训练负样本从固定目录均匀抽取，排除完整 training 行内商品，不读取 validation/test 行。全目录评价不删除冷启动目标，沿用其他模型的历史过滤语义。SASRec 保留官方默认商品历史长度50及自身超参数，不直接套用 LIGER 的 SID token 长度或训练目标；配置可覆盖，比较协议必须记录差异。

## 验证环境

原 `.venv` 的 PyTorch 为 `2.9.1+cpu`；独立临时环境使用 `2.9.1+cu128` 在本地 RTX4060 Laptop 完成 dry-run、5次 optimizer update、step5恢复以及16用户预测，未使用 node1。首次训练退出的 GBK 问题与 checkpoint betas 序列化问题及修复过程见 [基线说明](sasrec-baseline.md)。逻辑检查不能作为正式准确性结果，正式目录一致性、完整训练和完整评价仍由后续实验验证。
