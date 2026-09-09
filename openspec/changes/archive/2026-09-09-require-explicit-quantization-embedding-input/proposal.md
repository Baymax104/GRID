## Why

RKMeans、RVQ 和 RQVAE 的官方训练/推理脚本目前都将 `embedding_path` 固定为同一个历史 W&B run，导致脚本不可移植，并可能在未察觉的情况下消费过期或错误的 embedding。量化 experiment 已将该字段定义为必填人工输入，因此脚本入口也应显式暴露并校验它。

## What Changes

- **BREAKING**：六个量化训练/推理脚本不再提供硬编码的 `embedding_path`，调用者必须显式传入 `--embedding-path`。
- `--embedding-path` 同时支持 `--embedding-path=<value>` 和 `--embedding-path <value>`，并接受本地路径或 `wandb://` 引用。
- 缺失、空值或 separated syntax 缺少取值时，脚本以状态码 2 失败并输出明确错误。
- 将解析后的值作为单个 `embedding_path=<value>` Hydra override 传给统一入口，同时保持 `--dry-run`、训练 notes、推理 checkpoint 和尾部额外 Hydra override 的现有行为。
- 增加六个脚本的参数契约测试，并移除对固定 producer run `wandb://vb8es5ow` 的依赖。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `cli-dry-run`: 扩展官方量化 W&B 脚本契约，要求训练与推理入口显式接收 embedding 数据源并保持现有 CLI/Hydra 参数兼容性。

## Impact

- 受影响脚本：`rkmeans_train.sh`、`rkmeans_inference.sh`、`rvq_train.sh`、`rvq_inference.sh`、`rqvae_train.sh`、`rqvae_inference.sh`。
- 受影响测试：新增量化脚本参数解析与透传测试。
- 外部行为：所有通过上述脚本启动的命令都必须增加 `--embedding-path <local-path|wandb://run-id>`。
- 不修改 experiment/data 配置、Python launcher、Artifact resolver、W&B lineage 或 bundle 协议，也不新增依赖。
