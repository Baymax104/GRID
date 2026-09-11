# 训练方向探针人工执行方案

## 范围与对照

开发设置为 Beauty / RKMeans，SID `wandb://4vyi4o6w`，初始权重 `wandb://26qh50do`。只在用户手动启动后训练。原 checkpoint 为 frozen anchor；普通 CE 与 reweighted 两组均从该权重重新初始化 Adam，2000 个额外 optimizer steps，lr=5e-5，sequence length=180、beam=10、seed=42。

两组使用相同数据、文件版本、worker 数、设备数、batch 和精度。默认训练组件为每设备 batch=128；下面使用两张卡，与原训练的 global batch=256 一致。设备可由用户调整，但必须两组一致。这里的 seed 是 continuation seed，不是新的独立 baseline pretraining seed。

## 启动前检查

- 核验 `26qh50do` 的 checkpoint、SID 和数据 lineage 与当前文件一致。严格 state_dict 加载不能证明 SID 含义一致。
- 确认 training/evaluation/testing 分割不变。统计只读取 training，并记录 SHA256；统计口径为每文件完整遍历的期望展开目标，不是训练 2000 步实际看到的样本数。
- CE 和 reweighted 使用完全相同的统计和 alpha/cap/layers；只有 `probe_arm` 不同。权重表构建不消耗 RNG。
- `ckpt_path=null` 必须保持；初始化路径单独记录。不要用 `ckpt_path` 来实现本提案的继续训练。
- `run_test_after_training=false` 保持关闭；evaluation 已经用于方向开发，不能当未使用过的 held-out testing。
- `--dry-run` 可加到命令中进行统一入口 smoke；该模式仍可能读取初始化和统计数据，不能代替纯 compose 单元测试，也不作为正式实验。

## 两组人工训练命令

从 GRID 根目录执行，替换数据根目录；不要修改同一对实验中的共享条件。

```bash
NPROC_PER_NODE=2 bash tiger_training_probe.sh \
  --data-dir /path/to/beauty --devices '[0,1]' --group rkmeans \
  --semantic-id-path wandb://4vyi4o6w \
  --initialization-checkpoint-path wandb://26qh50do \
  --arm ce --seed 42 --master-port 29630 \
  --notes 'Training direction probe; Beauty RKMeans; fresh Adam; CE control; 2000 extra steps'

NPROC_PER_NODE=2 bash tiger_training_probe.sh \
  --data-dir /path/to/beauty --devices '[0,1]' --group rkmeans \
  --semantic-id-path wandb://4vyi4o6w \
  --initialization-checkpoint-path wandb://26qh50do \
  --arm reweighted --seed 42 --master-port 29631 \
  --notes 'Training direction probe; Beauty RKMeans; conditional branches L2 L3; alpha .25 cap 2; 2000 extra steps'
```

根脚本保留末尾 Hydra overrides。改变 max_steps 会同步改变 last checkpoint 保存步数。每组主比较必须用 finished run 在第 2000 步发布的 last checkpoint；中断、少步或来源不一致的 run 不进入主比较。checkpoint 附带 `training_frequency_probe`，其中包含 arm、初始化 URI、统计 manifest、SID 指纹和权重 lookup。W&B checkpoint Artifact metadata 同时记录初始化、SID、arm 与预算。

## 评估与比较

对两组最终训练 run 分别使用现有 `tiger_prefix_trace.sh`，`--ckpt-path wandb://<训练run>`、`--data-split evaluation`、`--beam-width 10`，其余 SID/数据/seed 一致。原 checkpoint 也使用相同 trace 配置作为 anchor；复用 `hho25h7i` 前核验 keys、split、配置及 Artifact 身份。

```bash
NPROC_PER_NODE=1 bash tiger_prefix_trace.sh \
  --data-dir /path/to/beauty --devices '[0]' --group rkmeans \
  --data-split evaluation --beam-width 10 --seed 42 \
  --semantic-id-path wandb://4vyi4o6w --ckpt-path wandb://REPLACE_TRAIN_RUN \
  --notes 'Training probe evaluation; replace with CE or reweighted arm and training run ID'

bash tail_sid_diagnosis.sh \
  --data-dir /path/to/beauty --seed 42 --group rkmeans \
  --semantic-id-path wandb://4vyi4o6w \
  --recommendation-output-path wandb://REPLACE_TRACE_RUN \
  --fixed-prefix-trace-path wandb://REPLACE_TRACE_RUN \
  --notes 'Training probe single-checkpoint diagnosis; evaluation only' \
  embedding_path=wandb://3jtt9mpa calibration_statistics_ready=true
```

替换所有占位符后人工启动。每个 checkpoint 独立 diagnosis；本提案没有新增跨训练 checkpoint 聚合器。不能使用 candidate-allocation paired analysis，因为该接口要求同一 checkpoint，且本干预没有开启解码配额。跨训练比较由结构化 evidence 与逐 user keys 的审阅完成。

必须记录：训练 run、实际 global step、初始化 URI、checkpoint Artifact、统计指纹、数据/seed/设备/batch、trace 和 diagnosis run。组别划分始终沿用训练原始 item 频次；训练权重使用期望目标频次，两种频次不得混用。

## 预声明方向门槛

1. 有效性：两个训练 run 均 finished 且恰为 2000 步，预算/来源/keys/split/SID 相同；唯一干预因素为训练目标。任一条件不满足为 inconclusive。
2. 机制：分别报告第 2、3 层 teacher target probability、legal rank 与 legal Top10 rate；第 2 层 Tail 平均 legal rank 降低且 Top10 rate 上升，作为主要评分改善证据。第 3 层不能隐藏反向变化。
3. 推荐：Tail + Cold Hit@10 相对 CE 和 anchor 均提高；Tail + Cold NDCG@10 不下降；Overall Hit@10 损失不超过 0.2 pp、Head 不超过 0.5 pp，分别相对 CE 和 anchor 检查。报告所有组新增/丢失命中，不只展示 Tail 相对增幅。
4. 三项均通过才允许扩展配对 continuation seeds；单 seed 只支持 provisional advance，不宣称因果或跨设置稳定。后续区间可使用配对 user/prefix cluster bootstrap，必须保持配对单位一致。
5. 评分改善但效用失败：该重加权版本 no-go。评分没有改善：当前强度/统计口径下的方向证据不足，不自动开启大范围 sweep。CE 本身明显退化时仍报告全部结果，不能只凭赢过 CE 判为成功。

完成该开发验证后再决定额外 seeds、训练预算和跨数据集实验，不自动开始原 18-setting 矩阵。
