# LIGER 预算匹配对照：完整结果与判断

## 结论

预算匹配对照已完成：LIGER 新增三段 6000/6000/2000 更新、两次完整单卡 Validation 和一次完整单卡 Testing，全部通过实际产物审计。双方累计训练与选择消耗均为 **64000 次 optimizer 更新、global batch 256、16384000 次扩展后训练样本呈现**，最终都使用两个 checkpoint 固定等权 logits pooling。

在 Beauty、seed 42 和相同 history 排除评价口径下，固定 CoPMRec 相对本次验证集选出的 LIGER baseline，Testing **Recall@10 提升 7.896221%，NDCG@10 提升 8.121248%**；两项配对绝对差值的 95% bootstrap 区间下界均为正。此前对旧 50k、单 checkpoint LIGER 的双 10.77% 数值仍是历史事实，**不能继续表述为预算匹配条件下双指标超过 10%**。

本次满足用户提出的第二种证明——匹配实际训练更新和样本呈现预算。LIGER 经续训与 pooling 后，Testing 对原 LIGER 的 Recall@10、NDCG@10 分别提升 2.663578%、2.455935%，因此不将原末期平台解释为“重启训练后已完全收敛”。该增量包含续训与 pooling 的共同效果，不作单组件因果归因。

机器可读最终证据见 [final-comparison.json](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/final-comparison.json)，唯一新 Testing 为 [a6moio4n](https://wandb.ai/baymaxam/GRID/runs/a6moio4n)。

## 问题、定位与冻结协议

用户明确要求“补一个预算匹配的 LIGER 对照，开始”。此前 CoPMRec 的唯一生产训练链为 50k+6k+6k+2k，共 64k 更新；原 LIGER 仅 50k，且单 checkpoint 部署。竞争解释是额外训练、fresh optimizer/LR 重启以及两个 checkpoint pooling 解释了部分旧收益。已有原日程末 5k 验证平台不足以排除该解释。

本次是 LIGER 原生方法的预算与部署对照，不提出新 CoPMRec 方法。冻结可反驳预测为：将 LIGER 补足相同实际更新预算和固定 pooling 机会后，CoPMRec 原双 10% 优势是否仍保留。负向结果按预登记收缩该主张，统计区间跨 0 则保留不确定性，不据此自动调参或扩预算。

- 从原 LIGER 自己的 Val best 45000 权重开始；各段只加载权重，并重建 AdamW 和 scheduler。原完整 50000 更新仍计入实际消耗，不以 best 45000 替代已花费预算。
- A/B 各 6000 更新，以原 dense、无 history 排除的 Val NDCG@10 选择自己的 best；C 共 2000 更新，以 history-excluded dense Val NDCG@10 选择自己的 best。
- 每段 AdamW LR 0.0001、weight decay 0.035、warmup 300、cosine horizon 6000、min ratio 0。C 在 2000 停止，horizon 保持 6000。
- 保留原 LIGER 的 SID CE 与 full-catalog content CE，权重各 1；训练 cold fill 为 -100，未加入 CoPMRec mixture、residual 或 bias。
- global batch 256、每卡 128、seed 42、FP32、clip 1、每 1000 更新验证；causal max 32、有效输入历史最多 20 商品、序列长度 80，与既有对照一致。
- 三段训练使用同型号 A100-SXM4-80GB：物理 GPU 5,6 经 CUDA_VISIBLE_DEVICES=5,6 映射本地 [0,1]。推理仅物理 GPU 5，经 CUDA_VISIBLE_DEVICES=5 映射本地 [0]。
- 最终候选为原 LIGER、C single、固定 A/C 各 0.5 pool。各成员独立 history query 与 catalog projection 后，平均完整 catalog logits；使用同有效输入 history 排除与 stable catalog row tie-breaking。
- 仅按完整 Validation 的 NDCG@10 → Recall@10 → 更简单方案冻结唯一 baseline，之后只运行一次新 Testing；固定 CoPMRec ws2fx4oi 输出直接复用。

## 实际训练与预算

| 阶段 | W&B run | 实际新增更新 | 自己的 best step | best Val NDCG@10 | 选择口径 |
| --- | --- | ---: | ---: | ---: | --- |
| A | [6an0pxdy](https://wandb.ai/baymaxam/GRID/runs/6an0pxdy) | 6000 | 6000 | 0.046695642 | 原 dense，无 history 排除 |
| B | [eqlhnpgt](https://wandb.ai/baymaxam/GRID/runs/eqlhnpgt) | 6000 | 6000 | 0.047325812 | 原 dense，无 history 排除 |
| C | [034h3uvv](https://wandb.ai/baymaxam/GRID/runs/034h3uvv) | 2000 | 2000 | 0.054184120 | history-excluded dense |

B 的原 dense 验证 NDCG@10 比原 LIGER best 约高 1.86%，支持原模型在非零 LR 重启后仍有训练改善。C 改变了验证资格口径，不能将 B→C 的差值当作纯训练收益。训练内 float32 scalar 与下表独立输出重算的数值存在浮点精度差异，最终比较使用独立重算值。

| 实际预算口径 | LIGER | 固定 CoPMRec |
| --- | ---: | ---: |
| 原训练完整更新 | 50000 | 50000 |
| 共享链新增更新 | 6000+6000+2000 | 6000+6000+2000 |
| 累计实际生产与选择更新 | **64000** | **64000** |
| global batch | 256 | 256 |
| 扩展后训练样本呈现 | **16384000** | **16384000** |
| 最终 pool 成员数／权重 | 2／0.5, 0.5 | 2／0.5, 0.5 |
| source 成员被选权重祖先更新 | 51000 | 54000 |
| continued 成员被选权重祖先更新 | 59000 | 62000 |

被选权重祖先不同，源于原 LIGER best45000、CoPMRec v0 best48000；各自原完整训练均消耗 50000。本次匹配的是已花费的更新与样本呈现，不为凑祖先长度替换各自 best，不把两个成员共享祖先重复累计为两条独立训练链。**不声称相同 FLOPs、GPU 实际工作量或全部超参数搜索预算。**

新增阶段预算严格用尽：3/3 训练、14000/14000 更新、2/2 完整 Validation、1/1 新 Testing。旧 CoPMRec 搜索阶段 7 次／34000 更新及旧 Testing 3/3 保留封存，未重置。三段 W&B runtime 共 1303 秒，分配双卡的墙钟估计为 0.723889 GPU 小时；该估计含验证、上传与等待，不能用作实际 GPU 活动或 FLOPs 测量。

![三段原生 LIGER 训练验证曲线](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/training-curves.png)

图中 A/B 与原 LIGER 使用同一验证口径；C 单独展示 history-excluded 口径，避免跨口径解释训练增量。[可导出 PDF](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/training-curves.pdf)。

## 完整 Validation 与预先冻结的选择

| 候选 | Validation run | Recall@10 | NDCG@10 |
| --- | --- | ---: | ---: |
| 原 LIGER，同 history 排除 | [wdms8w77](https://wandb.ai/baymaxam/GRID/runs/wdms8w77)，复用 | 0.096945848 | 0.054048899 |
| C single | [d649k83e](https://wandb.ai/baymaxam/GRID/runs/d649k83e) | 0.097527165 | 0.054184138 |
| **A/C 固定等权 pool** | [3or6e3p0](https://wandb.ai/baymaxam/GRID/runs/3or6e3p0) | **0.097974333** | **0.054519159** |

pool 按冻结规则胜出，选择记录 [validation-selection.json](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/validation-selection.json) 在 Testing 前保存，未读取 Testing 指标来选择模型、checkpoint 或 pool 权重。pool 对原 LIGER 的 Validation 增量为 R10 +1.060886%、N10 +0.870066%；两项配对差值 CI 跨 0，因此该 Validation 增量本身不能宣称显著。

最终 A/C checkpoint 均为各自 Val best，引用与实际 SHA256：

```text
A: wandb://baymaxam/GRID/6an0pxdy?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt
SHA256: 5d14cd3f51bc6501f1c1abf0908e4bf94d4f522bbfe7eb7933b2f8ccc15027e6

C: wandb://baymaxam/GRID/034h3uvv?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=002000.ckpt
SHA256: 9ea81ca478b8d6b4652ace612b6183b2972fb2e1560ff4d5b60e2c36e3a1cceb
```

## 唯一新 Testing 与剩余收益

| 模型／预算／部署 | Testing run | Recall@10 | NDCG@10 |
| --- | --- | ---: | ---: |
| 原 LIGER，50k，单成员，同 history 排除 | [vnhmag7v](https://wandb.ai/baymaxam/GRID/runs/vnhmag7v)，复用 | 0.077225775 | 0.042815584 |
| **预算匹配 LIGER，64k，A/C pool** | [a6moio4n](https://wandb.ai/baymaxam/GRID/runs/a6moio4n) | **0.079282744** | **0.043867107** |
| **固定 CoPMRec，64k，原固定 control pool** | [ws2fx4oi](https://wandb.ai/baymaxam/GRID/runs/ws2fx4oi)，复用 | **0.085543085** | **0.047429664** |

| 固定 CoPMRec 相对预算匹配 LIGER | LIGER | CoPMRec | 相对收益 | 配对绝对差值 95% CI |
| --- | ---: | ---: | ---: | --- |
| recall@5 | 0.053660063 | 0.058265886 | **+8.583333%** | [0.002504136, 0.006841658] |
| ndcg@5 | 0.035582556 | 0.038636381 | **+8.582364%** | [0.001713095, 0.004492923] |
| recall@10 | 0.079282744 | 0.085543085 | **+7.896221%** | [0.003800921, 0.008630327] |
| ndcg@10 | 0.043867107 | 0.047429664 | **+8.121248%** | [0.002318436, 0.004845236] |

统计区间使用同用户配对的 bootstrap，报告的是 CoPMRec−LIGER 的绝对指标差。相对收益按 CoPMRec/LIGER−1 重新计算，不以反向比较百分比取负替代。区间条件于本次固定 Beauty、seed42、已选择模型，未校正此前反复开发或推广为跨数据集／跨 seed 结论。

预算匹配 LIGER 对原 LIGER 的 Testing 增量分别为 R10 +2.663578%、N10 +2.455935%，其绝对差值 CI 为 [0.000670751, 0.003532621]、[0.000370681, 0.001724393]。因此旧 baseline 低估了经同预算续训与 pool 部署后的 LIGER，更新最终论文比较为本次更强 baseline。

作为剩余错误差异的描述，CoPMRec 对预算匹配 LIGER 新增 457 个 top10 命中，同时丢失 317 个，净增 140 个；共同命中用户中 527 个排名提高、451 个降低。NDCG@10 的新增、丢失与共同命中位置贡献分别为 +0.008133658、−0.005763160、+0.001192059。这些是结果分解，不据此证明某个机制的因果贡献或自动启动新的 bad-case 优化。

## 有效性与复现证据

- 两次新 Validation 与一次新 Testing 均独立校验各自 175 个原始文件、22363 用户、真实目标标签、完整 keys、catalog 与输入 Artifact；输出为唯一合法 top10，且不含有效输入 history 商品。Val/Test keys 恰好相同不能证明 split 一致，已另外验证原文件与标签 SHA。
- 三段实际 terminal 更新、完整验证历史、各自 Val best Artifact 的 selection=best、producer/digest、checkpoint 实际字节与递归来源均审计。163 个原生 state tensor、fresh optimizer 预检为空以及各阶段 152 个实际 AdamW 状态与 LR 日程一致。
- 所有新正式训练／推理的 runtime source 实际归档均为 383 个文件，聚合 SHA256 为 99412e402ee17bbbd7eed2aef0cb7c6eb743ee3652424f60633f9fba49dc06c3；原 374 个 runtime 文件逐字节不变。通过官方 Mutagen flush，三个 session Watching、无 conflict 后才交给远端。
- 106 项聚焦测试通过，覆盖原生损失／梯度等价、只权重恢复、pool 语义、Hydra compose、Bash 语法及 notes/quoting/空值/错误输入/override 透传；CPU preflight 与真实 DDP2 一步 smoke 通过。smoke 未建立 W&B formal run，未作为效果证据。
- 唯一新 Testing 的 W&B config 中 4 个冻结 Validation 指标 metadata 存在约 10⁻¹⁷ 的 JSON 浮点末位差。只对这 12 个可识别统计 scalar 允许绝对容差 10⁻¹⁵，保留差异路径、值与 hash 证明；模型、数据、source、notes、checkpoint 和选择规则仍精确比较，实际输出 Artifact metadata 的数值无差异。实际 0.0 与声明 0 按既有精确数值语义相等，不宣称 JSON 字节一致。只重做只读审计，未重跑正式 Testing。
- 新 runtime 归档不用于补写原始 50k 训练历史上缺失的 source archive。其历史 provenance 限制继续披露。

完整来源见 [三段训练 A](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/training-A.json)、[B](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/training-B.json)、[C](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/training-C.json)、[完整 single Val](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/inference-val-single.json)、[完整 pool Val](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/inference-val-pool.json) 和 [完整新 Test](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/inference-test-selected.json)。独立终审见 [final-review.json](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/final-review.json)。

实际自包含命令保存在同目录的 train-A.sh、train-B.sh、train-C.sh、val-single.sh、val-pool.sh、test-selected.sh，checkpoint 引用已固定；只用于复现归档，本阶段已执行完毕。

## 阶段判断

1. **核心假设**：匹配 LIGER 实际更新预算及固定 pool 机会后，CoPMRec 双指标 10% 优势仍成立。
2. **支持层次**：固定 CoPMRec 在当前匹配对照下仍有约 8% 的 R10/N10 正向收益，四项整体指标的配对绝对差值 CI 下界均正；保留该效果及当前部署。
3. **门槛结果**：双 10% 点估计均未通过，收缩“相同预算下超过 10%”主张；这不等于 CoPMRec 无效。原 LIGER 重启后的训练与部署改善也不支持完全收敛的说法。
4. **已缩小的不确定性**：64k 更新和样本呈现、同历史资格与双成员 pool 条件下的真实剩余收益已测得。相同 FLOPs／全部搜索成本、多 seed／其他数据集与各机制因果占比仍未知，本次不追加验证。
5. **成本与取舍**：本阶段 3 train/14k、2 Val、1 Test 全部用尽并封口；未追加扫描、训练或 Testing。最新记录同步为预算匹配下约 8% 收益，不从旧双 10% 门槛重新分配预算。
