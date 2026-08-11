## Context

GRID 当前通过 `sem_embeds_inference -> rkmeans_train -> rkmeans_inference -> tiger_train` 形成 TIGER 风格主链路。`sem_embeds_inference` 与 `rkmeans_inference` 的落盘产物均为 keyed prediction bundle，内容为 `{"keys": ..., "predictions": ...}`，由 `src.data.utils.load_model_output` 统一加载并按 key 排序。

第一个创新点需要在不改变训练链路的前提下，离线分析 `rkmeans_inference` 的 Semantic ID 产物，并输出长尾 item 的结构分辨率损伤证据。用户要求核心模块位于 `quantization/` 下的新增实验目录，运行脚本位于项目根路径，运行结果需要清晰展示。

## Goals / Non-Goals

**Goals:**
- 新增一个独立 CLI，可读取 `data_dir`、`semantic_id_path`、可选 `embedding_path` 并输出诊断结果。
- 在 `src/quantization/tail_sid_diagnosis/` 下实现核心逻辑，保持与量化实验域一致。
- 计算 P0 指标：频次分组、full collision、near collision、prefix density、suffix weakness、semantic mismatch、item damage、prefix risk。
- 以 `.json` 与 `.csv` 输出结果，便于直接查看和后续脚本读取。
- 提供根路径脚本 `tail_sid_diagnosis.sh`，使用 `uv run` 调用 CLI。
- 补充 CPU 单元测试覆盖 SID 标准化、频次分组、结构指标和输出生成。

**Non-Goals:**
- 不修改 RKMeans、RVQ、RQVAE 或 TIGER 模型训练逻辑。
- 不把诊断接入 Hydra `src/main.py`，第一版保持独立 CLI。
- 不实现 recommendation correlation、beam score 分析或生成重排。
- 不引入 parquet/pandas/matplotlib 等新依赖；第一版只输出可读表格与 JSON。

## Decisions

### Decision 1: 复用 keyed prediction bundle 作为主输入

诊断模块通过 `load_model_output` 读取 semantic ID 和 embedding。这样保持与 `keyed-prediction-bundle-artifact` 一致，不假设 item id 等于 tensor 行号。

Alternative considered: 直接 `torch.load` 裸 tensor 并按行号解释。该方式与当前 GRID 输出协议不一致，且会破坏非连续 key 支持。

### Decision 2: 独立 CLI 而非 Hydra experiment

第一版新增 `src.quantization.tail_sid_diagnosis.run` module 入口，由根脚本封装成 `uv run --module ...`。这样不会扩大统一入口和配置树的 blast radius。

Alternative considered: 新增 `experiment=tail_sid_diagnosis` 与 `run_mode=analysis`。这更统一，但需要扩展 launcher 分发和 Hydra 配置，超出第一阶段诊断核心。

### Decision 3: CSV/JSON 输出优先

输出 `summary.json`、`group_metrics.csv`、`item_damage_scores.csv`、`prefix_risk_scores.csv`。CSV 能直接查看，也不需要新增依赖；后续如果数据量过大，再在第二个提案中引入 parquet 或分片输出。

Alternative considered: 第一版直接输出 parquet。该方式更适合大表，但当前 `pyproject.toml` 未显式依赖 pandas/pyarrow，会引入额外环境成本。

### Decision 4: 结构指标先用 prefix bucket 计算

near-collision、density、suffix uniqueness 使用 prefix bucket 聚合，避免全量 pairwise O(N²)。Semantic mismatch 只在 strict deep prefix bucket 内采样计算，并保留全部 tail item。

Alternative considered: 全量 pairwise cosine 和 prefix overlap。该方式更直接，但 Beauty/Sports/Toys 全量运行风险过高。

### Decision 5: 输出清晰展示而非只落盘数据

CLI 结束时打印 summary 表、group metrics 摘要和输出文件路径。`summary.json` 同时包含关键组间指标，方便快速判断 tail damage 是否存在。

Alternative considered: 仅写文件。这样不满足“运行结果需要清晰展示”的要求，且不利于 smoke run 快速判断。

## Risks / Trade-offs

- [Risk] CSV 对大型 item/pair 产物不如 parquet 高效 -> 第一版默认不输出全量 pair 表，只输出 item 与 prefix 表；后续提案可升级存储格式。
- [Risk] 只用 semantic embedding 判断 mismatch 可能不覆盖协同相似性 -> 第一版将 collaborative compatibility 明确作为非目标，后续提案扩展。
- [Risk] 不接入 Hydra 会与训练链路配置风格不同 -> 根脚本和 argparse 参数保持与现有脚本一致，先降低主链路风险。
- [Risk] Damage 权重是启发式 -> 输出各原始指标和综合分数，允许论文实验中做敏感性分析，不把综合分数作为唯一证据。
