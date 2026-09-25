# CGBS 同初始化 B/C 训练结果

## 判定

2026-09-16 在线核验：B/C 正常完成，但 **当前单种子、20k steps 协议下，C 未超过内容初始化 A，未通过进入主方法确认阶段的总体指标门槛**。不能据此否定全部内容方法，也不能因为 C 超过 B 就认定在线分支必要。

当前只安排原定 screen 的 B/trained、C/trained、C/off 三次冻结推理，收集逐用户结果与真实路径；不追加训练、seed、Sports、超参数或全部 signal 队列。

## W&B 身份与训练完整性

| 条件 | run | best NDCG@10 | 对应 Recall@10 | best step | 最后5次 NDCG 均值 |
|---|---|---:|---:|---:|---:|
| A token_content_init | [5g3wpbg7](https://wandb.ai/baymaxam/GRID/runs/5g3wpbg7) | 0.042569727 | 0.080033332 | 19000 | 0.041577636 |
| B content_init_aux | [z98fozox](https://wandb.ai/baymaxam/GRID/runs/z98fozox) | 0.040364187 | 0.076055847 | 19000 | 0.039370181 |
| C content_init_full | [k6jvoo2v](https://wandb.ai/baymaxam/GRID/runs/k6jvoo2v) | 0.041390747 | 0.077491172 | 19000 | 0.040083681 |

本表全部使用训练期间相同 validation 口径。Recall@10 为最大 NDCG 所在记录的值，不是各自另选峰值；不要与独立推理结果混用。

- 三者均 finished，40次验证，最后 trainer/global_step=19999，即完成20k updates。
- seed42、devices=[0,1]、ckpt_path=null、非dry-run、beam10、sequence length180、SID层数与codebook一致。
- 完整 model.root 去掉 arm 后相等；train/val dataloader 相等；trainer 去掉输出路径后相等；checkpoint monitor 相等。没有发现预算或配置混杂。
- 三者使用同一 Artifact ID：`rkmeans_inference-semantic-id:v2`（`QXJ0aWZhY3Q6MzQ0MDQ2NjExMQ==`），`sem_embeds_inference-semantic-embedding:v5`（`QXJ0aWZhY3Q6MzQzOTg1ODUxMQ==`）。数据路径相同不等于已核验历史原始数据字节不变。
- B checkpoint：`tiger_catalog_grounded_content_init_aux_beauty_train-checkpoint:v0`，Artifact ID `QXJ0aWZhY3Q6MzYyNjQzMjIyNQ==`。
- C checkpoint：`tiger_catalog_grounded_content_init_full_beauty_train-checkpoint:v0`，Artifact ID `QXJ0aWZhY3Q6MzYyNjg1Mjc4Mw==`。
- 两个文件均为 `checkpoint_epoch=000_step=019000.ckpt`，metadata selection=best、monitor=val/ndcg@10，best_model_score 与曲线最大值一致。

原始 config、完整 history、summary、notes、checkpoint/used artifact 身份保存在 `tmp/cgbs_mechanism_training/runs.json`；只读采集代码 `tmp/inspect_cgbs_mechanism_training.py`。

## 机制解释的边界

- B−A：NDCG 相对 −5.18%。此辅助监督配置没有提高简单初始化模型的最终开发效果；不能只凭此数字断言梯度冲突是唯一原因。
- C−B：相对 +2.54%；C−A：相对 −2.77%。在线分支及其联合训练可能补回了 B 的部分损失，但这不是在强基线上获得净增益。
- 最后5次验证 C 均低于 A；15k到20k的11个配对检查点中，C只有1次高于A；C同期9次高于B。因此差异不只来自一个 checkpoint 峰值。
- 整段40次验证 C有22次高于A，主要在较早训练阶段；这不能转写成最终质量提升，也不足以证明固定质量下训练加速。
- W&B runtime：A1878s、B1935s、C2882s。C/A约1.535，但跨时段执行且包括验证/日志等开销，只能描述实际run耗时，不能当作受控训练吞吐或推理延迟结果。
- 尚无 B/C 逐用户独立推理、配对置信区间、Head/Tail、真实beam存活或C/off结果。因此不得宣布统计显著、长尾改善或在线分支的因果收益。

## 收尾推理命令

从 node1 仓库根目录依次执行，均用物理 GPU0；冻结 best checkpoint，不训练：

```bash
bash ./tiger_catalog_grounded_mechanism_inference.sh \
  --data-dir data/beauty --condition b --stage screen \
  --checkpoint-path 'wandb://baymaxam/GRID/z98fozox?role=checkpoint&file=checkpoint_*.ckpt' \
  --notes "CGBS B screen; frozen best step19000; compare with content-init A"

bash ./tiger_catalog_grounded_mechanism_inference.sh \
  --data-dir data/beauty --condition c --stage screen \
  --checkpoint-path 'wandb://baymaxam/GRID/k6jvoo2v?role=checkpoint&file=checkpoint_*.ckpt' \
  --notes "CGBS C screen; frozen best step19000; trained versus content-off"
```

第一条运行1次B/trained；第二条运行2次C/trained、C/off。目标是判断完整分支在同一checkpoint上的净作用，并与A/B逐用户对比。即使 C/trained 超过 C/off，只要仍不能超过 A，也不能恢复“复杂分支优于初始化”的主张。

这些对照完成后先作本机制的去留判断；只有出现明确指向近似误差的证据时才考虑既定 signal 干预，不因为结果为负就自动扩大实验。
