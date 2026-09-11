## Why

四 checkpoint Prefix Survival pilot 已得到 H2 No-Go：Tail 存活劣势存在，但未建立稳定的 competition 机制；K50 增加候选可达性也没有改善本轮 Tail Hit@10。当前 matched 指标混合了频次定义、用户并列选择和不同估计量的 CI，需要先可靠地区分候选不可达、候选内排序损失与逐层退出，再决定是否值得开展小规模干预。

## What Changes

- 在现有 `tail_sid_diagnosis` 链路中可选接收 widened recommendation bundle，配对 K10/K50 的完整 SID、user key、label、split 和 Artifact lineage。
- 输出互斥且完备的搜索—排序状态、Top10 新增/丢失命中、Hit/NDCG、候选可达率和固定候选集的 oracle 上界；避免把 K50 coverage 当成 Hit@10。
- 复用已存 trace，输出累计 survival、parent 存活条件下的退出率、首次失败 rank/margin 和缺失值支持数；保留 teacher rank 与真实 beam rank 的语义边界。
- 增加版本化的静态风险分层标准化估计：先聚合 item 内用户，限定共同支持，报告平衡与保留率，并用 prefix-cluster bootstrap 重算同一估计量。frequency 作为连续描述变量，不再声称在完全相同 frequency 下识别 Head/Tail 效应。
- 保留旧文件/字段含义，将旧 prefix matched/CI/verdict 显式标注为 legacy；新分析不得自动改写原 H2 No-Go。
- 使用现有 Beauty/Sports × RKMeans/RVQ × seed 42 的 evaluation Artifact 完成四组重分析，形成 `probe_candidate`、`stop` 或 `inconclusive` 的审阅记录。候选资格不是方法 Go。
- 仅分析与统计修正；不实现 calibration、改动解码分数、训练新 checkpoint、生成新 trace 或启动 18-setting 矩阵。

## Capabilities

### New Capabilities

- `tail-search-ranking-decomposition`: 定义配对候选状态、真实 Top10 效果、候选内 oracle 上界与后续探针资格评估。
- `tail-static-risk-standardization`: 定义 item 级静态风险共同支持、逐层标准化差距、频次描述与一致估计量的聚类区间。

### Modified Capabilities

- `tail-sid-diagnosis-evidence-artifact`: 增加可选输入身份审计、版本化分析文件和兼容的 summary/manifest 元数据。

## Impact

- 依赖当前尚未归档的 `add-tiger-prefix-survival-instrumentation` 实现与 trace schema；依据其 `pilot-analysis.md`、`pilot-audit.json` 的四组输入，保留其 No-Go 历史。实施前确认依赖代码可用，不要求本提案创建时归档旧提案。
- 数据侧涉及 `src/data/components/artifacts.py`、`data_models.py`、`src/data/datamodule/diagnosis.py`、`src/data/datasets.py`；统计侧涉及 `src/quantization/tail_sid_diagnosis/`。
- 配置和脚本涉及 `configs/{experiment,data,model}/tail_sid_diagnosis.yaml` 与 `tail_sid_diagnosis.sh`；继续使用 `src.main`、统一 launcher、`Trainer.test` 和共享 writer/lineage callback。
- 增加聚焦内存测试、Hydra compose、shell 参数验证；不引入新依赖，不修改标准 recommendation bundle。
- 本提案仅管理 GRID 中的实现与分析交付。研究仓库状态同步作为交接事项记录，不把外部文件列为本提案的实现目标。
