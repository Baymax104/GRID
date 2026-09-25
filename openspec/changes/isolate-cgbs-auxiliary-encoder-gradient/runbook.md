# CGBS 辅助梯度隔离：D 条件

最新状态：D 已完成并在线核验，best validation NDCG@10=0.036045112，相对 C 下降 12.915%。本说明保留原交付协议；当前结论与下一步命令以 [训练结果分析](training-outcome.md) 为准，下一步先补齐原 C 的冻结开关对照。

## 本次只回答一个问题

保持完整 CGBS 的内容初始化、前向打分、辅助损失权重和训练预算，去掉辅助 item CE 对共享编码器的直接更新，能否改善原 C？这是优化路径干预，不是已确认的梯度冲突结论。

```text
q_mix = query(encoded)
q_aux = query(stop_gradient(encoded))
loss = mixed_generation_CE(q_mix) + 0.1 * item_CE(q_aux)
```

两个 query 使用同一个 MLP。辅助 CE 仍训练 MLP；混合生成 CE 仍通过生成分支和内容分支更新编码器。新增 arm 为 `content_init_full_aux_detached`，原 C 的 arm 和 checkpoint 契约保留。

## 手动启动训练

在 node1 仓库根目录 `/data3/weizhenyu/projects/GRID` 执行：

```bash
bash ./tiger_catalog_grounded_mechanism_train.sh \
  --data-dir data/beauty \
  --condition d \
  --notes "CGBS refinement; auxiliary encoder gradient isolation only; same initialization and 20k budget as C"
```

脚本固定 `CUDA_VISIBLE_DEVICES=0,1`、`NPROC_PER_NODE=2`、`devices=[0,1]`，只启动一个 D。默认 seed42、20,000 optimizer steps、每 500 steps 验证、每卡 batch128（两卡总 batch256）、lr0.0005；保持原 best validation NDCG@10 checkpoint 选择。SID `wandb://4vyi4o6w`、embedding `wandb://3jtt9mpa` 与原 B/C 相同。额外 Hydra override 仍可放在命令末尾，但上述对照请使用原样命令。

W&B Config 应记录 `catalog_arm=content_init_full_aux_detached`、`mechanism_reference_run=k6jvoo2v`、`mechanism_baseline_run=5g3wpbg7`、`mechanism_revision=aux-encoder-stop-gradient-v1`。reference/baseline 是比较标记，实际输入 lineage 仍由已有 artifact callback 记录；D 从头训练，不加载 C checkpoint。

## 结果判读和下一道门

已有结果来自 `tmp/cgbs_mechanism_training/runs.json` 的 W&B 快照；本次尚无 D 训练结果：

| 条件 | W&B run | best validation NDCG@10 |
| --- | --- | ---: |
| A：内容初始化 | 5g3wpbg7 | 0.0425697267 |
| C：完整 CGBS、同内容初始化 | k6jvoo2v | 0.0413907468 |
| D：仅隔离辅助编码器梯度 | 待训练 | 待测 |

1. 首先核验 D 是否完成 20k steps、验证次数和上游 lineage 是否一致，比较完整 validation 曲线与 best checkpoint，而非只看最后一个 loss。
2. D 超过 C：支持这项路径干预在当前设置下有用。D 仍低于 A，则尚未解决复杂分支不及初始化的问题；D 超过 A 才出现完整模型的正向候选证据。任何极小差距或单 seed 均不能称显著或稳健。
3. 训练结果核验后，用同一 D best checkpoint 做 `trained/off`，区分在线内容路由贡献与训练表示变化。只有 `D trained > D off` 才获得此设置下在线内容分支有益的直接对照；还需结合 A/C、内容置换、成本和独立 seed 才能讨论论文结论。
4. D 未改善时保留该负结果，先根据已有冻结推理证据定位生成能力和内容贡献，不自动增加调参矩阵、门控或新假设。

后续 screen 模板如下；把 `<D_RUN_ID>` 替换为核验后的真实 ID 再执行。当前不要连带启动所有阶段。

```bash
bash ./tiger_catalog_grounded_mechanism_inference.sh \
  --data-dir data/beauty \
  --condition d \
  --stage screen \
  --checkpoint-path 'wandb://baymaxam/GRID/<D_RUN_ID>?role=checkpoint&file=checkpoint_*.ckpt' \
  --notes "CGBS D; frozen best validation checkpoint; paired trained versus off"
```

推理使用 GPU0，单进程；checkpoint 文件选择明确排除 `last.ckpt`。原 C checkpoint 不得用 D arm 恢复。

## 验证边界

CPU 测试覆盖同初始参数/损失/随机状态、辅助和生成梯度分解、D checkpoint round-trip、C/D 身份拒绝、推理干预、Hydra compose 与启动脚本契约。这些检查验证代码实现，不能代替 GPU/DDP 完整训练效果。

原研究评估提出的运行时梯度夹角/量级及独立纯 token loss，本次不加入训练循环：额外 decoder forward 会改变随机流，DDP 内部 autograd 探针需要单独验证。当前不声称已测得梯度冲突；相关机制测量保留为后续有界诊断，而非本轮新增实验矩阵。

## 交付验证记录（2026-09-16）

- `uv run pytest tests/recommendation/test_tiger_catalog_grounded.py tests/test_tiger_catalog_grounded_config_script.py -q`：147 passed；仅依赖弃用告警。
- 测试 hook 按 Ruff 要求绑定闭包变量后，两个梯度分解测试再次通过；模型和两个测试文件 Ruff check/format check 均通过。
- `openspec validate isolate-cgbs-auxiliary-encoder-gradient --strict`：通过。
- `scripts/mutagen_sync.ps1 flush` 成功；后续完整 status 核验 `grid-src`、`grid-configs`、`grid-scripts`、`grid-root-code` 均为 One Way Replica、两端已连接、Watching for changes，无 conflict。仅同步代码受管范围，文档和测试在本地。
- 未启动完整训练、推理或诊断；D 结果及 GPU/DDP 实验验证待用户执行。
