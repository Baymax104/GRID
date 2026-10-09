# BRIR 收尾与 Sports 内容初始化对照

执行后状态：本轮四个audit与Sports训练均已完成并核验，结论及下一步手动推理命令见[结果报告](results.md)。下文保留启动前协议与当时的W&B快照。

## 当前决策

暂停 BRIR v1 主方法扩展。下一轮只做四个已完成 checkpoint 的有限审计，以及一个缺失的 Sports `token_content_init` 训练。后者用于确定简单内容基线，不代表已确定新的论文主方法。

2026-09-16 只读核对 W&B：Sports `mask_ce` run `lhevwroq` 已完成，GPU devices=[0,1]、每卡 batch128、无梯度累积、20k steps、每500步验证、seed42、ckpt_path=null、run_test_after_training=false。查询 `dataset_name=sports AND catalog_arm=token_content_init` 返回空，缺失对照尚未运行。

## 1. 手动运行 BRIR 有限审计

在 node1 的 GRID 仓库根目录执行。四个任务串行，共用 sampling_seed=42 选取的128个 evaluation 用户，仅使用GPU0。根脚本仍调用已有 `brir_audit.sh` → `src.main experiment=brir_audit`。

```bash
NPROC_PER_NODE=1 bash ./brir_training_diagnostics_suite.sh \
  --data-dir data/beauty \
  --gpu 0 \
  --notes "BRIR closure; frozen last checkpoints; candidate versus catalog; evaluation 128 users; no new training"
```

可先追加 `--print-only` 查看四条实际命令，不加载模型、不创建run。`--dry-run` 会执行统一入口的有界 smoke 逻辑，不能作为正式结果。单个任务失败时，可追加 `--arm dense`（或base、prefix_free、brir）重跑对应任务，避免重复成功的run。

| arm | 完成训练run | checkpoint选择 |
|---|---|---|
| base | `ffa9bc2e` | `?role=checkpoint_last`，20k |
| dense | `iv1h5zky` | `?role=checkpoint_last`，5k |
| prefix_free | `2ux40teb` | `?role=checkpoint_last`，5k |
| brir | `82t0rv4c` | `?role=checkpoint_last`，5k |

公共来源：Beauty SID `wandb://4vyi4o6w`、embedding `wandb://3jtt9mpa`；分支初始化来自base run `ffa9bc2e`，标定来自 `wandb://py8ovqwa?role=brir_calibration`。不使用best与last之间有歧义的引用。

每个run通过现有 writer 发布 role=`brir_audit` 的 `brir_audit.pt`，schema=`brir_audit_v2`。checkpoint、上游来源和正式条件由 W&B Config 与 lineage 记录。默认候选是目标+64 hard+16 random；base上的候选CE是共同候选集反事实，base真实训练使用全目录CE。

## 2. 审计结束后，手动补 Sports 训练

两项任务都使用GPU0，因此按顺序运行。沿用现有 CGBS20k协议，从头训练；物理GPU为0、1，有效batch=128×2=256，固定seed42。

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash ./tiger_catalog_grounded_train.sh \
  --data-dir data/sports \
  --dataset sports \
  --group rkmeans \
  --semantic-id-path wandb://3narllqy \
  --embedding-path wandb://psec3u5i \
  --devices '[0,1]' \
  --arm token_content_init \
  --seed 42 \
  --notes "Missing Sports content-init control; matched CGBS 20k protocol; seed42; effective batch256; GPUs0,1; no test-based selection"
```

本轮不追加BRIR训练、多个seed或超参数扫描。Sports训练完成后，需按原有CGBS流程使用验证集选择的best checkpoint做同协议keyed evaluation推理，才能与既有推理结果对比。不要直接将训练过程的最高标量与旧的独立推理结果混成主结果表；本轮尚无新run ID，不预填其下游引用。

## 3. 结果判读顺序

1. 先核对四组run已完成，产物行数128，keys、input_sha256、labels、catalog、anchor_fingerprint一致；比较完整anchor分数与候选keys，浮点分数允许微小容差。
2. 检查 `dynamic_exact_set_match`、`dynamic_exact_order_match` 和原始bound。若与reference不一致，先处理搜索或数值问题；reference本身不改善时，增加搜索预算不能修复评分。
3. 对dense同时比较 `anchor_candidate_ce`→`candidate_ce`、`anchor_full_ce`→`full_ce` 与真实目标排名。候选外概率质量、Top-K候选外数量和hard-negative重合用于定位竞争来源。全目录CE天然不小于候选CE，二者存在差值本身不构成失败证据。
4. 对残差组比较base与final目标排名、目标残差、饱和比例；`bounded_best_target_rank>10` 表示在该query的base评分及当前delta下，即便每个item可独立选择最有利残差，也无法把目标送入Top10。该乐观界忽略共享网络约束，不意味着界内排名一定可达。
5. CPU测试验证了从同一base初始化的三个未更新分支评分与候选一致。GPU证据中的base分数仅为当前模型的零残差反事实；dense的初始化参照应看冻结anchor。
6. 将这些观察与Sports简单内容对照一起用于下一次方法决策。128用户不能替代全量evaluation、独立seed或显著性检验；本轮不根据test选择方法。

## 4. 开发验收

聚焦测试覆盖零更新初始化、真实候选截断（128-item目录/81候选）、候选CE与全目录排序反向变化、残差饱和、标签不改变预测、坏证据拒绝、双用户writer合并、Hydra compose、脚本quoting及override透传。完整GPU链路由上述手动命令验证。
