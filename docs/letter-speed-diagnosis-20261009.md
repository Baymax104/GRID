# LETTER 运行速度排查（2026-10-09）

> 修复已应用并同步 node1；六个真实 checkpoint、402 用户的有限差分核验通过，当前空闲 GPU 核验约加速 4.7–5.0 倍。见 [修复与验证记录](letter-prefix-fix-20261009.md)。下文保留初次排查的证据与环境。

> 2026-10-09 更新：六组推荐训练已停止，所对应的 W&B run 与六个源码 Artifact 已按用户要求删除。本文运行状态与链接为历史记录；清理核验见 [停止与清理记录](letter-cancel-20261009.md)。CF、Tokenizer、SID 上游保留。

## 结论

存在前缀约束生成的性能实现缺陷，主要耗时集中在validation。此次只做只读检查和独立有界计时，不改模型/配置，不重启六组正式训练，不启动Testing。诊断进程均已退出。

## 运行证据

精确读取六个W&B run的history _step=[0,65)，取得每组前五次validation。连续train/loss记录通常间隔50 optimizer step；最后训练记录至validation指标的gap包含验证及边界开销，不是Trainer直接计时。

| Run | 50训练步中位秒 | validation gap中位秒 | 验证时间占比估计 |
| --- | --- | --- | --- |
| [vzzddvhg](https://wandb.ai/baymaxam/GRID/runs/vzzddvhg) | 2.675 | 1049.0 | 97.5% |
| [oremp4gb](https://wandb.ai/baymaxam/GRID/runs/oremp4gb) | 2.698 | 1178.4 | 97.8% |
| [od6uvwdi](https://wandb.ai/baymaxam/GRID/runs/od6uvwdi) | 2.735 | 1175.2 | 97.7% |
| [h628mv5l](https://wandb.ai/baymaxam/GRID/runs/h628mv5l) | 2.762 | 1926.4 | 98.6% |
| [gzi64g6z](https://wandb.ai/baymaxam/GRID/runs/gzi64g6z) | 2.604 | 1967.5 | 98.7% |
| [xzf4m2f0](https://wandb.ai/baymaxam/GRID/runs/xzf4m2f0) | 2.702 | 1918.7 | 98.6% |

占比估计公式：validation_gap/(validation_gap+10×train_50_gap)。六组均满足validation_gap大于500训练步耗时的10倍；取数时六组均running。

## 根因与排除项

src/recommendation/letter/backbone.py:107每次对CUDA prefix执行tolist()。安装的transformers5.14.1 PrefixConstrainedLogitsProcessor逐batch/beam调用回调，并用Python token list逐行写CUDA mask索引。batch32×beam20×五步=3200次回调，引入小规模D2H/H2D搬运和同步。

真实Beauty/Sports evaluation batch [32,81]，实际保存的推荐checkpoint及固定SID strict加载。GPU7 profiling的单次生成：encoder执行1次、decoder执行5次且输入长度1/use_cache=true，prefix回调3200次、cudaStreamSynchronize9635次、D2H pageable copy3201次。故“重复encoder/未启用cache”未成立。profile-results.json的forward列表还记录后续计时阶段，不能用列表长度代替单次profile decoder计数。

GPU共享影响绝对耗时，但不能解释同GPU同输入仅替换前缀查询后的一致大幅改善；此次没有独占GPU对照，不能量化共享因素的独立贡献。

## 不加instrumentation的有界对照

物理GPU0、FP32/medium、现有共享负载；去除profiler、hooks和计时回调，仅用CUDA synchronize+perf_counter。每dataset固定checkpoint和一个32用户evaluation batch，三次交替原实现与临时批量Prefix LogitsProcessor。保留原HF generate、beam20/top10、五步、EOS、length_penalty、tie排序和全目录约束。

临时原型每步整批prefix一次tolist()，按相同字典规则构造整批索引，一次scatter写mask。只替换独立进程中的模型实例，未写入项目实现或现有训练。

| Dataset | 原实现中位秒 | 批量原型中位秒 | 倍数 | 输出 |
| --- | --- | --- | --- | --- |
| beauty | 4.4533 | 0.1122 | 39.7× | 三次商品ID与分数均torch.equal |
| sports | 4.4397 | 0.1153 | 38.5× | 三次商品ID与分数均torch.equal |

共两个真实batch、64用户，支持性能机制和有界输出等价；不证明全量DDP等价、全量validation加速倍数或最终基线有效性。GPU7 instrumentation统计与GPU0正常计时不能混合比倍数。原始checkpoint路径/SHA256、逐次耗时和profile统计保存在logs/letter-speed-diagnosis-20261009/*.json；node1同目录保留两个Chrome trace。

## 修复方向

在独立LETTER模块中实现批量前缀约束LogitsProcessor，保留HF beam search及概率/EOS/目录语义；之后可考虑GPU Trie索引。保留完整用户集合、beam20与每500步验证协议。覆盖多batch及EOS/padding/同分边界，完成模块级OpenSpec和聚焦等价验证后再同步。

运行中的进程不会因同步自动更新代码。切换修复需要明确恢复checkpoint与新run/source记录；不能假定last.ckpt是最新step或数据流可无损恢复。

另有较小冗余：forward将labels传给T5（内部计算CE）后再次手工计算温度CE。未单独计时；训练阶段很快，因此不作为本次主要慢路径根因。

