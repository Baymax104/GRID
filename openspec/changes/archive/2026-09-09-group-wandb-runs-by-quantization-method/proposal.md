## Why

当前 W&B run 按单个任务 family 分组，导致同一种量化方法的量化训练、semantic ID 推理、TIGER 训练/推理和 Tail-SID diagnosis 分散在多个 group 中，难以按端到端实验链路比较。需要将 group 重新定义为量化方法，并让 run name 明确展示任务阶段与时间。

## What Changes

- 将 `rkmeans`、`rvq`、`rqvae` 定义为三条端到端量化方法 group；各自包含量化 train/inference、消费对应 semantic ID 的 TIGER train/inference 和 Tail-SID diagnosis。
- 保留公共语义向量生产任务的独立 `sem_embeds` group。
- **BREAKING**：`tiger_train`、`tiger_inference` 和 `tail_sid_diagnosis` 不再默认使用任务 family group，必须显式提供 `group=rkmeans|rvq|rqvae`。
- 为三个对应启动脚本增加必填 `--group` 参数，同时兼容 `--group=value` 与 `--group value`，并继续保留额外 Hydra override 的尾部覆盖能力。
- 将所有官方 W&B run name 统一为 `${task_name}/${now_tz:%Y-%m-%d_%H-%M-%S}`；`job_type` 继续区分 train、inference 和 analysis。
- 不改变 Artifact 名称、Artifact lineage、Hydra run id 或输出目录布局。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `experiment-config-componentization`: 将 W&B group 从任务 family 调整为量化方法对应的端到端 pipeline group，并规定统一 run name 格式。
- `cli-dry-run`: 扩展官方 W&B 脚本契约，使 Tiger 和 diagnosis 脚本显式接收并验证量化方法 group，同时保持 dry-run、notes 和额外 override 兼容性。

## Impact

- 受影响配置：`configs/experiment/{tiger_train,tiger_inference,tail_sid_diagnosis}.yaml` 以及 `configs/logger/*.yaml`。
- 受影响脚本：`tiger_train.sh`、`tiger_inference.sh`、`tail_sid_diagnosis.sh`。
- 受影响测试：Hydra compose/W&B identity 测试、Tail-SID diagnosis 配置测试和启动脚本参数契约测试。
- 外部行为：通过官方 Tiger/diagnosis 脚本启动时必须新增 `--group`；直接使用统一 Hydra 入口时必须提供顶层 `group` override。
- 不新增依赖，不修改模型、数据、Artifact bundle 或 W&B Artifact resolver 协议。
