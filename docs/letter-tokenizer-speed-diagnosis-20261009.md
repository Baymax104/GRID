# LETTER Tokenizer 耗时排查（2026-10-09）

> 后续修复已应用并同步 node1。六个真实 checkpoint 的输出、梯度、随机状态及两步 optimizer continuation 均核验通过；生产修复有限测量加速3.46–4.27倍。见 [修复验证记录](letter-tokenizer-diversity-fix-20261009.md)。下文保留初次诊断证据。

## 结论

存在 Diversity 正样本采样的性能实现缺陷：每个 batch 按样本逐项读取 CUDA group label，四层 batch1024 共触发 4136 次 CUDA 标量转整数。有限真实 checkpoint 对照中，批量读取 labels/ids 后，前向加反向中位耗时从 117.8 ms 降至 22.4 ms，约 5.25 倍，五次不同固定随机种子的全部返回值与参数梯度均逐位相同。

耗时还受到正式预算影响：20000 epoch 是实际数据目录的 20000 次遍历，Beauty 每 epoch12步，共24万步；Sports 每 epoch18步，共36万步。这不是20000 optimizer step。当前六组均已完整结束。本轮不修改项目模型/配置，不停止或重启正式训练，不发布新 W&B run/Artifact；批量原型仅存在于独立有限诊断进程。

## 真实 run 时间

| Dataset | Run / seed | runtime秒 | 小时 | 已记录 step / epoch |
| --- | --- | ---: | ---: | --- |
| Beauty | [025s9ets / 42](https://wandb.ai/baymaxam/GRID/runs/025s9ets) | 37119 | 10.31 | 239999 / 19999 |
| Beauty | [6e7ruv7a / 200](https://wandb.ai/baymaxam/GRID/runs/6e7ruv7a) | 38079 | 10.58 | 239999 / 19999 |
| Beauty | [5m1w7c9r / 2026](https://wandb.ai/baymaxam/GRID/runs/5m1w7c9r) | 37792 | 10.50 | 239999 / 19999 |
| Sports | [h5912m4c / 42](https://wandb.ai/baymaxam/GRID/runs/h5912m4c) | 52570 | 14.60 | 359999 / 19999 |
| Sports | [6vzov2iw / 200](https://wandb.ai/baymaxam/GRID/runs/6vzov2iw) | 52448 | 14.57 | 359999 / 19999 |
| Sports | [lkrugs0y / 2026](https://wandb.ai/baymaxam/GRID/runs/lkrugs0y) | 52577 | 14.60 | 359999 / 19999 |

W&B step/epoch 为零起始记录。每组前70条 history 的相邻50训练步中位 gap 为6.65–7.48秒，包含跨 epoch 的分组与边界开销，不能当成纯 GPU forward 计时。runtime 是整个 run 的时间，不是 profiler GPU 时间。

作者固定 commit 的 [main.py](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/RQ-VAE/main.py) 默认 epochs20000、batch1024、eval_step2000、sk_iters50；[trainer.py](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/RQ-VAE/trainer.py) 每个 epoch 对各层 codebook 重新 constrained K-means 分组。当前预算和分组频率与这些设置一致，不应将降预算/降低分组频率伪装成等价性能修复。

## 可复现慢路径与有限差分

位置：`src/quantization/letter/tokenizer.py` 的 `diversity_loss()`。默认训练未传入 positives，会构造 group 字典，再对1024个 ids 执行 `int(labels[index])`。labels 为 CUDA tensor；四层4096次逐项读取，加上40个 group ID 转整数，产生4136次 CUDA scalar 转整数。

实际 Beauty seed42、metric-selected checkpoint 为 node1 `logs/letter_tokenizer_train_beauty/runs/2026-10-04/19-12-42/checkpoints/step_step=024000.ckpt`，SHA256 `7ad82fcbaf82402bacd7c15c34bc0a36e8f69ae238a158b2488bf4708608db9d`。加载真实共同内容与自身 CF，按目录取前1024商品。物理 GPU4 → CUDA0，A100 80GB，Torch2.9.1+cu128、FP32/medium，存在共享 GPU 负载。

baseline.json 的一步包含 forward/backward/AdamW更新，耗时124.7 ms；临时计数发现4136次 CUDA `__int__`，`scalar_read_regression_guard_passed=false`，明确复现逐样本同步问题。

正常对照不启用 profiler/计数 hook、不更新参数。仅临时替换 Diversity 采样，批量 `labels.tolist()` 和 `ids.tolist()`，按相同顺序构造 choices，保留 Python random.choice 的序列及原 loss。seed42–46 五次交替原实现/批量原型，所有返回 tensor 和参数梯度均 `torch.equal`。原 forward+backward 为111.4–124.0 ms，批量为20.4–29.4 ms，中位117.8/22.4 ms。

独立 profiler 的一个原始训练步记录：aten::item4155次、cudaStreamSynchronize4243次、DtoH pinned4195次、DtoH pageable44次、nonzero40次。`aten::item` CPU total 约88.6 ms。包含嵌套事件，不能把各项 CPU 时间直接相加；profiler 数据不用于计算正常耗时加速倍数。

## 其他候选瓶颈

- 每 epoch 四层分组：固定 checkpoint codebook、seed123、三次有限计时，n_jobs10 中位174.1 ms；首次调用1.04秒包含进程池启动。n_jobs1 中位729.2 ms，更慢；四层 centers、labels 和分组 membership 均与并行结果不同。因此不建议将 n_jobs 改为1；初次对照的等价断言失败保留为负向证据，后续探针分别记录差异。
- 当前分组成本累积20000 epoch约58分钟，仅为固定 checkpoint 的粗略外推；运行中的 codebook 和共享 CPU/GPU 负载变化，不能用它精确拆分历史 run。
- 固定输入上的50轮 Sinkhorn约4.7–6.4 ms；encoder/decoder各约0.4–0.5 ms（forward）；完整Beauty目录的 encoding+collision_rate约4.1–5.9 ms，不含数据搬运、Trainer、writer。这些有限结果不支持验证是十小时级主因，不能将它们直接累加成完整 step。
- 只有每2000 epoch验证一次，共10次。真实初始设置和 K-means 初始化只在启动时执行；没有发现它们在每个 batch 重复的调用路径。

## 修复建议与边界

最小修复是在独立 LETTER 模块中批量读取采样所需 ids/labels，保留 group 成员顺序、随机调用、loss 和梯度语义。不能换成不同的采样策略后声称逐位等价。更深的组缓存必须在 update_groups 和 checkpoint load 后保持一致，需要额外回归验证。

保留20000 epoch、每 epoch分组、n_jobs10、Sinkhorn50轮、FP32/medium与当前 best 选择规则。5.25倍仅为一个Beauty checkpoint/batch的 forward+backward差分，不是完整训练加速承诺；尚未验证Sports/Toys、完整 epoch、optimizer 状态演化或多机器等价性。

原始 JSON 位于本地与 node1 的 `logs/letter-tokenizer-speed-20261009/{run-timings,baseline,profile-results}.json`；node1 另留 `beauty-step-trace.json`。项目 tokenizer SHA256 `083640b3b0b444d7a1f2d249e32805b5f0e207193c5862d51c5147d0341a2714`，诊断未修改源文件。
