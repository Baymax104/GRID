## Why

当前 TIGER/RKMeans pipeline 只能产出 item embedding 与 Semantic ID，缺少一个离线诊断能力来证明长尾 item 是否被压缩到结构低分辨率 SID 区域。第一个创新点需要先形成可复用的诊断信号，再服务后续量化修复与生成重排。

## What Changes

- 新增 Tail-SID Resolution Damage 诊断能力，运行在 `rkmeans_inference` 之后、`tiger_train` 之前。
- 在 `src/quantization/` 下新增实验目录，承载 SID 读取、频次分组、结构指标、damage score 与展示输出。
- 新增项目根路径运行脚本，使用 `uv run` 启动单数据集诊断。
- 输出清晰的 `summary.json`、`group_metrics.csv`、`item_damage_scores.csv` 与 `prefix_risk_scores.csv`，用于人工检查和后续模块复用。
- 不修改 TIGER 训练、RKMeans 推理或现有 keyed prediction bundle 协议。

## Capabilities

### New Capabilities
- `tail-sid-resolution-diagnosis`: 定义长尾 SID 分辨率损伤诊断的输入契约、核心指标、输出产物与运行展示。

### Modified Capabilities
- `keyed-prediction-bundle-artifact`: 诊断模块复用现有 keyed prediction bundle，不改变既有要求。
- `independent-quantization-models`: 诊断模块新增在 `src/quantization/` 下，但不改变各量化模型自包含要求。

## Impact

- Affected code: `src/quantization/tail_sid_diagnosis/`, `tests/quantization/tail_sid_diagnosis/`, root `tail_sid_diagnosis.sh`.
- Affected specs: 新增 `tail-sid-resolution-diagnosis` capability。
- APIs: 新增独立 CLI module；不改变现有 Hydra experiment 和 model/data 配置。
- Dependencies: 不引入新依赖；第一版使用 `.csv` 而非 parquet，避免扩大环境成本。
