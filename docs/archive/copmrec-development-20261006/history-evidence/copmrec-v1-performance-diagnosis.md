# CoPMRec v1 训练性能排查（2026-10-03）

## 结论

v1 的持续慢速主要来自候选排序目标的 decoder 前向和反向计算，以及逐用户、逐分块执行带来的大量 kernel 启动；在线 beam 搜索也是显著开销。GPU2 同时有另一训练进程实际使用大量算力，是当前运行的额外干扰。短窗口的数据读取不足以解释持续慢速。

本次仅做观测和零更新诊断，没有修改运行代码、训练配置、候选集合、评价协议或当前生产进程，也没有启动新的正式训练、创建 W&B run 或 checkpoint。

## 运行与时间窗口

- 当前 v1：[uzmkrfoa](https://wandb.ai/baymaxam/GRID/runs/uzmkrfoa)，Beauty，seed 42，2 卡，单卡 batch 128，排序用户 4，候选分块 64，FP32，首次 validation 在 2500 更新。
- 历史 v0：[7y54j4m6](https://wandb.ai/baymaxam/GRID/runs/7y54j4m6)，取早期训练窗口作为吞吐线索。
- 从 W&B 的 `_runtime` 与 `trainer/global_step` 差值计算：v1 的 54 个有效 50 更新间隔中位数为 **0.3839 秒/更新**，范围 0.3747–0.6987；历史 v0 的 37 个有效间隔中位数为 **0.0990 秒/更新**。
- 排除 global step 未增加的 validation 行和验证后的首个间隔。历史数据来自有界 history 查询，并非完整曲线；硬件负载和评价协议不同，约 3.88 倍差异不能作为受控性能基准。
- 当前 run 在 global step 2499 的两个记录间隔为 **109.44 秒**，涵盖首次 validation 及关联回调。训练阶段在此前已出现慢速，不能全部归因于 validation。

原始返回及查询范围见 [wandb-history.json](evidence/copmrec-v1-performance-20261003/wandb-history.json)。

## 同批真实数据、预热后的零更新计时

在 node1 的物理 GPU1 上，用当前 run 的 `.hydra/config.yaml` 装配模型与训练 dataloader。训练流保留实际 preprocessing、单 rank 文件分片、4 个 worker、128 batch、80 SID token、12101 商品。该卡当时已有进程占用显存，但观测的计算利用率为 0%；它不是受控独占环境。

比较同一个随机初始化 v1 模型上的 v0 基础目标和 v1 完整目标。每个目标预热 3 次、计时 10 次，包含前向与反向；没有 optimizer step、DDP 通信或 optimizer state。未加载当前训练 checkpoint，因此不能覆盖已训练模型在不同候选分布下的全部开销。

| 路径 | 前向加反向中位数 | 范围 |
| --- | ---: | ---: |
| v0 基础目标 | 77.93 ms | 77.07–86.90 ms |
| v1 完整目标 | 346.04 ms | 343.70–371.88 ms |

在这一受限、同批比较中，v1 为 v0 的约 **4.44 倍**。新增排序路径能解释大部分持续慢速，不需要以数据加载或 DDP 异常作为解释前提。

随后对 5 次 v1 更新做边界 CUDA synchronize 计时；该插桩会改变调度，因此以下用于定位，不能直接相加重构未插桩吞吐。

| 前向阶段 | 每次更新平均耗时 |
| --- | ---: |
| history encode | 13.12 ms |
| catalog projection | 4.82 ms |
| 基础目标 `_joint_losses` | 16.17 ms |
| 在线 beam 搜索 | 46.19 ms |
| 构建 4 个用户的候选列表 | 1.48 ms |
| 4 个用户候选评分合计 | 81.84 ms |
| 排序目标整体，含 beam 和候选评分 | 131.18 ms |

每个排序用户实测 68–74 个候选，均超过 `candidate_chunk_size=64`。当前 `ranking_loss` 逐用户调用 `candidate_scores`，每用户再分为两个 chunk，因此一次更新中有 **8 次候选 decoder 前向及其反向**，外加基础目标 decoder 和 4 层 SID 的 beam 调用。候选 decoder 会为每个候选处理完整 SID，并在 cross-attention 中重复处理同一用户的 encoder 状态；`expand` 本身只是视图，不意味着后续投影和注意力计算得到复用。

源码位置：`src/recommendation/liger/relevance.py` 的 `candidate_scores`、`ranking_loss`、`retrieve`；beam 位于 `src/recommendation/liger/module.py::_generate_candidate_rows`。

两次插桩更新的 profiler 记录到 `cudaLaunchKernel` **29440 次**，仅该项 CPU self time 合计约 197.89 ms；它不是端到端吞吐时间。这支持大量小算子和重复 decoder 调用具有显著 host 调度成本，不能简单把低 GPU 利用率解释为数据加载阻塞。

原始计时、候选数量、profiler 摘要见 [warm-profile.json](evidence/copmrec-v1-performance-20261003/warm-profile.json)。`candidate_counts` 前 20 个数对应阶段计时，后续为 profiler 和评价调用；评价计时沿用了带同步的插桩，不作为独立吞吐基准。

独立重复探针从相同配置的 dataloader 获取另一批真实数据，仍为同批 v0/v1 比较，各预热 3 次、计时 10 次。在 forward/backward 边界分别 synchronize：

| 路径 | 前向中位数 | 反向中位数 |
| --- | ---: | ---: |
| v0 基础目标 | 34.05 ms | 51.85 ms |
| v1 完整目标 | 168.25 ms | 183.95 ms |

插桩统计每次更新 decoder **13 次**：基础目标 1 次、beam 4 次、候选评分 8 次。所有 decoder 前向合计约 119.10 ms，relevance MLP 前向合计仅约 **1.09 ms**。新增耗时落在候选 decoder 路径及其反向，MLP 本身不是主要成本。这里的 backward 是整个目标的反向时间，没有按模块单独归因。

拆分数据见 [forward-backward.json](evidence/copmrec-v1-performance-20261003/forward-backward.json)。两次探针的实际 inline 源码保存在同目录 `warm-probe-source.txt` 和 `phase-probe-source.txt`，用于追溯诊断条件，不是新增训练入口。

## 数据与共享 GPU

连续读取 25 个训练 batch：首批 131.56 ms，排除前 4 批后的读取中位数 **0.359 ms**，最大 **27.75 ms**。这只覆盖短窗口，不能排除文件切换时的长尾，但不足以支持数据读取是持续 0.38 秒/更新的主因。

`nvidia-smi pmon` 连续 5 次进程采样确认：

- GPU2 的 CoPMRec rank0（PID 2014774）SM 16%–28%。
- 同卡另一 `ray::MegatronTr` 训练进程（PID 2019438）SM **70%–82%**，占用约 53 GB 显存。
- GPU3 的 CoPMRec rank1（PID 2014778）SM 61%–70%；同卡 `sglang::scheduler` 的该窗口采样没有给出 SM 使用数值。

GPU2 的共享计算负载会干扰两个 rank 的速度平衡；尚未通过训练内 collective profiler 量化 DDP 等待，也没有建立历史慢速段与其他进程活动的一一时间对应关系。不能把所有慢速或全部长尾都归因于共享 GPU。

采样见 [gpu-process-sample.json](evidence/copmrec-v1-performance-20261003/gpu-process-sample.json)。本次没有终止或迁移任何进程。

## 优化优先级

1. **候选评分及反向。** 将不同用户的候选整理为批量输入，减少当前 8 次串行 decoder 调用；评估复用每个用户的 cross-attention K/V。保持候选集合、完整 SID 表示、最终 score 和梯度通路。任何实现都需要核对 dropout 与梯度语义以及显存预算。
2. **在线 beam 路径。** 定位 HF beam 与 prefix processor 的细分成本，优先减少冗余 prefix 索引和 host/device 同步。保留 beam20 与质量混合规则；不要通过关闭 beam 或沿用旧模型候选来冒充等价加速。
3. **运行资源。** 后续对比应使用计算负载可控的 GPU，才能分离模型改善与资源竞争。当前 run 保持运行。
4. **评价吞吐。** 首次 validation 约 109 秒，评价同样逐用户调用候选 decoder，可受益于评分批量化。暂不降低用户数量、验证频率或候选宽度。

此前 `production-probe.json` 中的 v0 0.827 秒、v1 0.587 秒为未预热的单次顺序测量，只能证明可执行与显存边界，不能用于判断相对训练速度；本次预热后的计时取代它作为性能诊断依据。
