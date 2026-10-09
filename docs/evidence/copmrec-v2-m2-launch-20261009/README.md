# CoPMRec v2 M2 训练启动回执

启动日期：2026-10-09（Asia/Shanghai；node1日志使用UTC）。用户授权M2九个正式训练实验，在node1各使用独立tmux，允许共享有多余显存的GPU。全部九个run已经W&B回读确认running并记录实际训练步数；此回执只验证启动，不表示训练完成或正式效果有效。

## Run与资源

| Issue | Dataset | Seed | W&B run | tmux | 物理GPU → local | torchrun端口 |
| --- | --- | --- | --- | --- | --- | --- |
| BMX-120 | beauty | 42 | [rmgfscr0](https://wandb.ai/baymaxam/GRID/runs/rmgfscr0) | bmx-120-v2-train-rmgfscr0 | 0,1 → [0,1] | 29840 |
| BMX-121 | beauty | 200 | [2cxk6qj9](https://wandb.ai/baymaxam/GRID/runs/2cxk6qj9) | bmx-121-v2-train-2cxk6qj9 | 0,1 → [0,1] | 29841 |
| BMX-118 | beauty | 2026 | [afbmzinx](https://wandb.ai/baymaxam/GRID/runs/afbmzinx) | bmx-118-v2-train-afbmzinx | 0,1 → [0,1] | 29842 |
| BMX-119 | sports | 42 | [hq166ou9](https://wandb.ai/baymaxam/GRID/runs/hq166ou9) | bmx-119-v2-train-hq166ou9 | 2,3 → [0,1] | 29843 |
| BMX-15 | sports | 200 | [8a7r4ys6](https://wandb.ai/baymaxam/GRID/runs/8a7r4ys6) | bmx-15-v2-train-8a7r4ys6 | 2,3 → [0,1] | 29844 |
| BMX-16 | sports | 2026 | [7doxj00o](https://wandb.ai/baymaxam/GRID/runs/7doxj00o) | bmx-16-v2-train-7doxj00o | 2,3 → [0,1] | 29845 |
| BMX-17 | toys | 42 | [jn3iqsep](https://wandb.ai/baymaxam/GRID/runs/jn3iqsep) | bmx-17-v2-train-jn3iqsep | 4,5 → [0,1] | 29846 |
| BMX-18 | toys | 200 | [bhj57tye](https://wandb.ai/baymaxam/GRID/runs/bhj57tye) | bmx-18-v2-train-bhj57tye | 4,5 → [0,1] | 29847 |
| BMX-19 | toys | 2026 | [hnkajj6r](https://wandb.ai/baymaxam/GRID/runs/hnkajj6r) | bmx-19-v2-train-hnkajj6r | 4,5 → [0,1] | 29848 |

每个实验使用 `copmrec_train.sh` → `uv run torchrun --nproc_per_node=2 -m src.main experiment=copmrec_train`，从仓库根目录执行；scratch、global256、FP32、50k保持不变。group为 `paper_main_copmrec_<dataset>`，name为 `copmrec_v2_<dataset>_seed<seed>_train/<原时间id>`，实际resolved config明确run ID、物理卡、v2与formal。三目标等权，无native view、全阶段无历史排除。

输出目录为 `logs/copmrec_v2/bmx-<issue number>_train_<run id>`。launcher/log位于node1 `logs/_launch/copmrec_v2_m2_20261009/<run id>/{launch.sh,launch.log,launch.json}`；结束后launcher将写入exit_code。tmux设置remain-on-exit保存终止状态。可用 `ssh node1` 后 `tmux attach -t bmx-120-v2-train-rmgfscr0` 查看对应会话。

## 启动核验

先核对GPU余量、现有任务、磁盘、数据目录、端口和同步。每组先启动seed42，确认每进程显存约3.7–4.0GiB后再启动其余六run。最终各run均有两个活跃GPU进程；GPU0–5仍余约17–44GiB，6–7未分配新训练。原任务没有被终止或修改。

执行了 `./mutagen_sync.ps1 flush`，三个session均Watching for changes/no conflict。本地及node1共288文件的源码SHA256一致：

```text
3e486ce5d7a206c13bee21f108d1e806f862712a3cc1afd5a6535bcd06bd8a00
```

九个run各有独立 `grid-source-<run id>:v0` code artifact，metadata中的source SHA匹配且origin.status=verified。已核对W&B真实resolved config中的版本、seed、模型、三目标、无排除、group、物理卡与预算。九个run的train loss均为有限值、global step均大于零；它们不作为推荐效果结论。

训练/evaluation/testing分片已在启动前计算文件SHA256，node1 manifest为 `logs/_launch/copmrec_v2_m2_20261009/dataset-shards.json`；[本地JSON副本](dataset-shards.json)保存逐文件路径、大小、SHA与各split规范化manifest摘要。SID/content仍使用冻结版本，不改变baseline或上游。

## 当前记录与执行边界

M2父任务及九个实验issue已经In Progress，命令替换为实际资源分配并登记训练run、来源、tmux与端口；[当前issue回读](issues-after.json)、[当前Run Registry](../../../../research/docs/copmrec-v2-run-registry-20261009.md)、research-state.yaml与当前计划已同步。

本次只启动M2九次训练（计划累计450k），不启动独立Testing、M3训练或诊断，不添加自动持续监控。训练完成后仍需审计自身首次最高raw val/ndcg@10的own-best checkpoint和完整来源，再进行单卡Testing；当前best URI/SHA、Testing指标与有效性判定保持null，没有标Done。

证据：[精确命令和资源](launch-plan.json)、[node1启动检查](startup-final.json)、[W&B真实config/summary/code artifact](wandb-startup.json)。执行调度脚本只调用现有根入口，不改变模型、配置、数据或虚拟环境。
