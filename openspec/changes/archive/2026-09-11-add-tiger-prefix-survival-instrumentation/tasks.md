## 1. Runtime 输出契约

- [x] 1.1 为 `ModelOutput` 增加默认空的 named auxiliary payload，并补充旧式 `keys + predictions` 构造兼容测试。
- [x] 1.2 定义 target-centric Prefix Trace tensor schema、字段 shape、dtype、1-based rank 与 `-1` sentinel，并为 schema validation 添加内存单元测试。
- [x] 1.3 定义 trace-enabled TIGER prediction 组装逻辑，使 recommendation predictions 与 auxiliary trace 共享同一 user keys 且互不改变。

## 2. TIGER teacher-forcing 与 beam instrumentation

- [x] 2.1 从现有 global teacher-forcing logits 计算逐层 target token probability、合法候选 rank 和 target-vs-best-legal margin。
- [x] 2.2 在 `TigerDecoder` constrained beam step 中只读采集 target prefix survival、beam rank、parent rank、target/cutoff score、cutoff margin 和合法候选数。
- [x] 2.3 汇总逐层 trace 并计算 1-based `first_failure_depth`，完整存活时写入 `-1`。
- [x] 2.4 增加 tracing disabled/enabled 输出逐元素一致测试，证明 instrumentation 不改变 generated IDs 或路径分数。
- [x] 2.5 增加 tiny-catalog golden tests，覆盖首层/中间层退出、完整存活、parent beam 重建、tie rank、非法 prefix 和不同 beam width。

## 3. Prefix Trace 本地与 W&B Artifact

- [x] 3.1 实现 domain-neutral auxiliary tensor shard writer，在 rank zero 按业务 key 合并并拒绝重复 key 或 shape 不一致。
- [x] 3.2 写入版本化 `prefix_trace.pt`，包含 `schema_version`、`keys`、`labels`、`trace` 和完整 source metadata。
- [x] 3.3 确认标准 Local/W&B prediction writers 忽略 auxiliary payload，且 `merged_predictions_tensor.pt` 仍只有 `keys` 与 `predictions`。
- [x] 3.4 实现 logger-owned W&B Prefix Trace Artifact 发布，设置 role/type `prefix_trace`，并保持无 W&B 的本地模式可用。
- [x] 3.5 通过共享 Artifact resolver 支持本地路径与 `wandb://` Prefix Trace 引用，并补充 role selection、manifest、publication failure 和 lineage 测试。

## 4. Split-safe trace experiment 与启动契约

- [x] 4.1 新增薄 TIGER prefix trace experiment/data/model/callback 配置，复用统一 `src.main`、launcher 和 `Trainer.predict`。
- [x] 4.2 将 `data_split` 声明为 mandatory `evaluation|testing` 输入，按值读取对应目录并写入 trace metadata；非法或缺失值必须提前失败。
- [x] 4.3 在 trace collate 中同时保留 `TigerLabelData.target_ids` 与 `TigerModelInput.output_keys`，验证 user ID 不进入模型特征。
- [x] 4.4 暴露 runtime `beam_width` override 并验证复用同一 checkpoint 时 K=10/K=50 均可 compose 和 dry-run。
- [x] 4.5 新增根目录 `tiger_prefix_trace.sh` 启动脚本，支持 `--data-split`、`--beam-width`、`--seed`、`--master-port`、`--devices`、`--group`、`--ckpt-path`、`--semantic-id-path`、`--notes`、`--dry-run` 与额外 Hydra override，并验证两种 flag 形式、引号、空值和 shell 语法。

## 5. Diagnosis 机制证据

- [x] 5.1 扩展 diagnosis data contract，以配置字段 `fixed_prefix_trace_path`、`widened_prefix_trace_path` 显式接收 fixed/widened Prefix Trace；同步为 `tail_sid_diagnosis.sh` 增加 `--fixed-prefix-trace-path`、`--widened-prefix-trace-path` 参数，并按 user keys、labels、split、checkpoint 和 semantic-ID identity 严格对齐。
- [x] 5.2 计算 layer × popularity group 的 teacher probability/rank、prefix survival、first-failure 和 cutoff-margin 汇总。
- [x] 5.3 增加 static-risk matched survival gap、prefix competition 增量关联和代表性设置的 prefix-cluster bootstrap。
- [x] 5.4 在 compatible widened trace 可用时计算 K=10 miss 到 K=50 recovery rate、failure-depth shift 和分组 coverage recovery。
- [x] 5.5 扩展 structured evidence 文件与 `summary.json` metadata，增加独立 H2 mechanism verdict，同时保持现有 structural/generation verdict 原义。
- [x] 5.6 增加结构化配置 `calibration_statistics_ready`；其为 `true` 时拒绝使用 `data_split=testing` 的 trace，并增加明确的数据泄漏错误测试。

## 6. 聚焦验证

- [x] 6.1 运行 decoder、TIGER runtime、writer/artifact 和 Tail-SID diagnosis 的聚焦 pytest。
- [x] 6.2 对 evaluation/testing、K=10/K=50、本地/W&B、trace on/off 组合执行 Hydra compose 或等价轻量验证。
- [x] 6.3 运行完整 `uv run pytest`，检查现有 recommendation bundle、metrics、Artifact lineage 和普通 inference 无回归。
- [x] 6.4 运行 `openspec validate add-tiger-prefix-survival-instrumentation --strict` 并修复全部验证错误。

## 7. 四 checkpoint 机制 pilot

- [x] 7.1 在 `data/beauty/evaluation` 上复用 checkpoint `wandb://26qh50do` + RKMeans SID `wandb://4vyi4o6w`，分别运行 K=10/K=50 trace 与 diagnosis。
- [x] 7.2 在 `data/beauty/evaluation` 上复用 checkpoint `wandb://ye9u9yj7` + RVQ SID `wandb://d2hhqdic`，分别运行 K=10/K=50 trace 与 diagnosis。
- [x] 7.3 在 `data/sports/evaluation` 上复用 checkpoint `wandb://129w8p0r` + RKMeans SID `wandb://3narllqy`，分别运行 K=10/K=50 trace 与 diagnosis。
- [x] 7.4 在 `data/sports/evaluation` 上复用 checkpoint `wandb://49ote174` + RVQ SID `wandb://ykntf4ve`，分别运行 K=10/K=50 trace 与 diagnosis。
- [x] 7.5 核对八条 trace run 的 checkpoint/SID/split/key lineage、Artifact schema、invalid prefix 和运行开销。审计通过；运行成本仅有 W&B runtime，未隔离 instrumentation 自身开销，详见 `pilot-analysis.md` 与 `pilot-audit.json`。
- [x] 7.6 汇总四组 risk-matched layer-wise survival、first failure、widened-beam recovery 与 competition-vs-frequency 证据，形成 H2 Go / No-Go 结论。2026-09-11：No-Go；明确区分逐层原始证据与当前四层平均 matched 指标，记录 matched CI 缺失及统计限制。
- [x] 7.7 仅当 H2 获得跨设置机制支持时创建后续 `add-budget-aware-prefix-decoding` 提案；否则记录失败模式并停止主方法实现。已执行 No-Go 分支，不创建主方法提案。

## 8. 用户启动命令

以下命令是 7.1–7.4 的可执行运行手册。必须先完成并验证 1–6；在此之前，`tiger_prefix_trace.sh` 及新增 diagnosis 参数尚不存在，不应提前运行。所有命令均从仓库根目录执行，默认使用单卡 GPU 0、evaluation split 和 seed 42。

### 8.1 生成八条 Prefix Trace run

- [x] 8.1 依次执行以下八条命令，并确认每条 run 都成功发布 recommendation output 与 `prefix_trace` Artifact：

```bash
NPROC_PER_NODE=1 MASTER_PORT=29610 ./tiger_prefix_trace.sh \
  --data-dir data/beauty \
  --data-split evaluation \
  --beam-width 10 \
  --seed 42 \
  --devices '[0]' \
  --group rkmeans \
  --ckpt-path wandb://26qh50do \
  --semantic-id-path wandb://4vyi4o6w \
  --notes "H2 pilot; Beauty RKMeans; evaluation; seed=42; beam=10; fixed"

NPROC_PER_NODE=1 MASTER_PORT=29611 ./tiger_prefix_trace.sh \
  --data-dir data/beauty \
  --data-split evaluation \
  --beam-width 50 \
  --seed 42 \
  --devices '[0]' \
  --group rkmeans \
  --ckpt-path wandb://26qh50do \
  --semantic-id-path wandb://4vyi4o6w \
  --notes "H2 pilot; Beauty RKMeans; evaluation; seed=42; beam=50; widened"

NPROC_PER_NODE=1 MASTER_PORT=29612 ./tiger_prefix_trace.sh \
  --data-dir data/beauty \
  --data-split evaluation \
  --beam-width 10 \
  --seed 42 \
  --devices '[0]' \
  --group rvq \
  --ckpt-path wandb://ye9u9yj7 \
  --semantic-id-path wandb://d2hhqdic \
  --notes "H2 pilot; Beauty RVQ; evaluation; seed=42; beam=10; fixed"

NPROC_PER_NODE=1 MASTER_PORT=29613 ./tiger_prefix_trace.sh \
  --data-dir data/beauty \
  --data-split evaluation \
  --beam-width 50 \
  --seed 42 \
  --devices '[0]' \
  --group rvq \
  --ckpt-path wandb://ye9u9yj7 \
  --semantic-id-path wandb://d2hhqdic \
  --notes "H2 pilot; Beauty RVQ; evaluation; seed=42; beam=50; widened"

NPROC_PER_NODE=1 MASTER_PORT=29614 ./tiger_prefix_trace.sh \
  --data-dir data/sports \
  --data-split evaluation \
  --beam-width 10 \
  --seed 42 \
  --devices '[0]' \
  --group rkmeans \
  --ckpt-path wandb://129w8p0r \
  --semantic-id-path wandb://3narllqy \
  --notes "H2 pilot; Sports RKMeans; evaluation; seed=42; beam=10; fixed"

NPROC_PER_NODE=1 MASTER_PORT=29615 ./tiger_prefix_trace.sh \
  --data-dir data/sports \
  --data-split evaluation \
  --beam-width 50 \
  --seed 42 \
  --devices '[0]' \
  --group rkmeans \
  --ckpt-path wandb://129w8p0r \
  --semantic-id-path wandb://3narllqy \
  --notes "H2 pilot; Sports RKMeans; evaluation; seed=42; beam=50; widened"

NPROC_PER_NODE=1 MASTER_PORT=29616 ./tiger_prefix_trace.sh \
  --data-dir data/sports \
  --data-split evaluation \
  --beam-width 10 \
  --seed 42 \
  --devices '[0]' \
  --group rvq \
  --ckpt-path wandb://49ote174 \
  --semantic-id-path wandb://ykntf4ve \
  --notes "H2 pilot; Sports RVQ; evaluation; seed=42; beam=10; fixed"

NPROC_PER_NODE=1 MASTER_PORT=29617 ./tiger_prefix_trace.sh \
  --data-dir data/sports \
  --data-split evaluation \
  --beam-width 50 \
  --seed 42 \
  --devices '[0]' \
  --group rvq \
  --ckpt-path wandb://49ote174 \
  --semantic-id-path wandb://ykntf4ve \
  --notes "H2 pilot; Sports RVQ; evaluation; seed=42; beam=50; widened"
```

### 8.2 回填 Prefix Trace run ID

- [x] 8.2 从上述八条 W&B run 页面复制 8 位 run ID，替换下列占位值并在同一 shell 会话中执行；`wandb://<run-id>` 将由共享 resolver 按 Artifact role 分别解析 recommendation output 与 Prefix Trace：

```bash
export BEAUTY_RK_K10_TRACE_RUN="m4h0geda"
export BEAUTY_RK_K50_TRACE_RUN="56rrarps"
export BEAUTY_RVQ_K10_TRACE_RUN="r94ut5sv"
export BEAUTY_RVQ_K50_TRACE_RUN="k7mr3xlk"
export SPORTS_RK_K10_TRACE_RUN="5024wy48"
export SPORTS_RK_K50_TRACE_RUN="4x3steyq"
export SPORTS_RVQ_K10_TRACE_RUN="lzttakbs"
export SPORTS_RVQ_K50_TRACE_RUN="n6svfy9g"
```

### 8.3 运行四组配对 diagnosis

- [x] 8.3 确认 8.2 中没有残留 `REPLACE_WITH_RUN_ID` 后，依次执行以下四条命令；每条 diagnosis 使用 K=10 recommendation output 作为结果基准，同时配对 K=10 fixed trace 与 K=50 widened trace：

```bash
./tail_sid_diagnosis.sh \
  --data-dir data/beauty \
  --seed 42 \
  --group rkmeans \
  --semantic-id-path wandb://4vyi4o6w \
  --recommendation-output-path "wandb://${BEAUTY_RK_K10_TRACE_RUN}" \
  --fixed-prefix-trace-path "wandb://${BEAUTY_RK_K10_TRACE_RUN}" \
  --widened-prefix-trace-path "wandb://${BEAUTY_RK_K50_TRACE_RUN}" \
  --notes "H2 paired diagnosis; Beauty RKMeans; evaluation; seed=42; beam=10 vs 50" \
  embedding_path=wandb://3jtt9mpa \
  calibration_statistics_ready=true

./tail_sid_diagnosis.sh \
  --data-dir data/beauty \
  --seed 42 \
  --group rvq \
  --semantic-id-path wandb://d2hhqdic \
  --recommendation-output-path "wandb://${BEAUTY_RVQ_K10_TRACE_RUN}" \
  --fixed-prefix-trace-path "wandb://${BEAUTY_RVQ_K10_TRACE_RUN}" \
  --widened-prefix-trace-path "wandb://${BEAUTY_RVQ_K50_TRACE_RUN}" \
  --notes "H2 paired diagnosis; Beauty RVQ; evaluation; seed=42; beam=10 vs 50" \
  embedding_path=wandb://3jtt9mpa \
  calibration_statistics_ready=true

./tail_sid_diagnosis.sh \
  --data-dir data/sports \
  --seed 42 \
  --group rkmeans \
  --semantic-id-path wandb://3narllqy \
  --recommendation-output-path "wandb://${SPORTS_RK_K10_TRACE_RUN}" \
  --fixed-prefix-trace-path "wandb://${SPORTS_RK_K10_TRACE_RUN}" \
  --widened-prefix-trace-path "wandb://${SPORTS_RK_K50_TRACE_RUN}" \
  --notes "H2 paired diagnosis; Sports RKMeans; evaluation; seed=42; beam=10 vs 50" \
  embedding_path=wandb://psec3u5i \
  calibration_statistics_ready=true

./tail_sid_diagnosis.sh \
  --data-dir data/sports \
  --seed 42 \
  --group rvq \
  --semantic-id-path wandb://ykntf4ve \
  --recommendation-output-path "wandb://${SPORTS_RVQ_K10_TRACE_RUN}" \
  --fixed-prefix-trace-path "wandb://${SPORTS_RVQ_K10_TRACE_RUN}" \
  --widened-prefix-trace-path "wandb://${SPORTS_RVQ_K50_TRACE_RUN}" \
  --notes "H2 paired diagnosis; Sports RVQ; evaluation; seed=42; beam=10 vs 50" \
  embedding_path=wandb://psec3u5i \
  calibration_statistics_ready=true
```

- [x] 8.4 核对四条 diagnosis run 的 resolved config 和 Artifact lineage：dataset、seed、group、checkpoint、SID、embedding、split 以及 fixed/widened trace identity 必须与 7.1–7.4 一致。
