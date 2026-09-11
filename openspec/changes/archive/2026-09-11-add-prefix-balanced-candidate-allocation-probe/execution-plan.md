# Prefix-Balanced Candidate Allocation Probe 人工执行计划

## 执行边界

- 本文件中的 inference、diagnosis 和 W&B 发布命令只由用户人工启动。
- 自动验证不得执行本文件中的命令。
- 首轮 intervention 固定为 `reserved_slots=1`、`pool_multiplier=2`、beam width 10、seed 42。
- baseline 使用同一 probe 入口、同一 pool multiplier，并显式关闭 allocation；这样新 trace schema 与运行身份完整，同时 generation 应复现既有 K10 输出。
- 若 Beauty RKMeans 首对结果没有任何 candidate set 变化，先审阅实现和 trace，不直接扩到另外三个设置。

## 注册设置

| 设置 | data_dir | group | checkpoint | semantic ID | embedding |
|---|---|---|---|---|---|
| Beauty RKMeans | `data/beauty` | `rkmeans` | `wandb://26qh50do` | `wandb://4vyi4o6w` | `wandb://3jtt9mpa` |
| Beauty RVQ | `data/beauty` | `rvq` | `wandb://ye9u9yj7` | `wandb://d2hhqdic` | `wandb://3jtt9mpa` |
| Sports RKMeans | `data/sports` | `rkmeans` | `wandb://129w8p0r` | `wandb://3narllqy` | `wandb://psec3u5i` |
| Sports RVQ | `data/sports` | `rvq` | `wandb://49ote174` | `wandb://ykntf4ve` | `wandb://psec3u5i` |

这些引用来自已核验的四个正式 search-ranking diagnosis run；运行前仍应确认本地数据目录和 W&B 登录状态。

## 单设置顺序

以下以 Beauty RKMeans 为模板。每一步完成后记录 W&B run ID；baseline 和 intervention run 都应各自发布 recommendation output 与 Prefix Trace Artifact。

### 1. Baseline

```bash
./tiger_prefix_allocation_probe.sh \
  --data-dir data/beauty \
  --data-split evaluation \
  --beam-width 10 \
  --devices '[0]' \
  --group rkmeans \
  --ckpt-path wandb://26qh50do \
  --semantic-id-path wandb://4vyi4o6w \
  --allocation-enabled false \
  --reserved-slots 0 \
  --pool-multiplier 2 \
  --notes 'Beauty RKMeans candidate allocation baseline; seed 42; beam 10'
```

### 2. Intervention

```bash
./tiger_prefix_allocation_probe.sh \
  --data-dir data/beauty \
  --data-split evaluation \
  --beam-width 10 \
  --devices '[0]' \
  --group rkmeans \
  --ckpt-path wandb://26qh50do \
  --semantic-id-path wandb://4vyi4o6w \
  --allocation-enabled true \
  --reserved-slots 1 \
  --pool-multiplier 2 \
  --notes 'Beauty RKMeans prefix-balanced candidate allocation; reserve 1; pool 2; seed 42'
```

### 3. 配对 diagnosis

将 `<BASELINE_RUN_ID>` 与 `<INTERVENTION_RUN_ID>` 替换为前两步的 W&B run ID。recommendation 与 trace 路径使用相同 run ID，由字段 role 解析到对应 Artifact。

```bash
./tail_sid_diagnosis.sh \
  --data-dir data/beauty \
  --group rkmeans \
  --semantic-id-path wandb://4vyi4o6w \
  --candidate-allocation-probe \
  --baseline-recommendation-output-path wandb://<BASELINE_RUN_ID> \
  --intervention-recommendation-output-path wandb://<INTERVENTION_RUN_ID> \
  --baseline-prefix-trace-path wandb://<BASELINE_RUN_ID> \
  --intervention-prefix-trace-path wandb://<INTERVENTION_RUN_ID> \
  --notes 'Beauty RKMeans matched candidate allocation diagnosis; baseline versus reserve 1 pool 2' \
  embedding_path=wandb://3jtt9mpa
```

## 其余三设置替换表

按单设置顺序分别运行 baseline、intervention、diagnosis，只替换下表字段：

| 设置 | data_dir | group | checkpoint | semantic ID | diagnosis embedding |
|---|---|---|---|---|---|
| Beauty RVQ | `data/beauty` | `rvq` | `wandb://ye9u9yj7` | `wandb://d2hhqdic` | `embedding_path=wandb://3jtt9mpa` |
| Sports RKMeans | `data/sports` | `rkmeans` | `wandb://129w8p0r` | `wandb://3narllqy` | `embedding_path=wandb://psec3u5i` |
| Sports RVQ | `data/sports` | `rvq` | `wandb://49ote174` | `wandb://ykntf4ve` | `embedding_path=wandb://psec3u5i` |

## 每对 run 的完整性检查

配对 diagnosis 前必须确认：

1. 两个 run 均为 `finished`，各有一个 recommendation output 和一个 Prefix Trace Artifact。
2. checkpoint、semantic ID、evaluation keys、labels、beam width 10、seed 42、SID shape 完全相同。
3. baseline metadata 为 `enabled=false, reserved_slots=0, pool_multiplier=2`。
4. intervention metadata 为 `enabled=true, reserved_slots=1, pool_multiplier=2, source_split=training`。
5. trace 含 target prefix mass、shortlist membership、reserve retention 和 actual reserve count。
6. diagnosis evidence manifest 保存五个输入 Artifact 的原始 URI、resolved path、producer run、artifact name/version/type/role/file。

## 决策门槛

单设置 evidence 只报告 setting gate；四设置全部完成后再计算总 verdict。

### `advance`

- 至少 3/4 设置的 Tail+Tail-Cold candidate access delta 大于 0；
- 至少 2/4 设置出现真实 Tail+Tail-Cold Top10 新增；
- 每个设置 overall Hit@10 下降不超过 0.2 pp；
- 每个设置 Head Hit@10 下降不超过 0.5 pp。

### `stop`

- 四设置完整，但未满足全部 advance 条件；特别是 Tail survival 或 candidate access 改善而没有真实 Top10 新增时，不进入完整矩阵。

### `inconclusive`

- 少于四个设置、任一配对身份审计失败、Artifact lineage 不完整，或结果无法区分 allocation 变化与输入变化。

若 verdict 为 `advance`，下一 change 再定义 Toys 与多 seed 扩展。若为 `stop`，保留 diagnosis 结论并转向训练目标或概率校准，不继续增加 allocation sweep。
