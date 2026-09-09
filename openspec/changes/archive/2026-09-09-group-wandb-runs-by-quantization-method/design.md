## Context

官方 experiment 在顶层声明 `group`，各 logger component 通过 `${group}` 配置 W&B。量化 train/inference 已分别使用 `rkmeans`、`rvq`、`rqvae`，但 TIGER 和 Tail-SID diagnosis 仍按自身任务 family 分组，使同一量化方法的端到端结果无法在一个 W&B group 中浏览。所有 logger 的 run name 当前只有时间戳，也不便于在混合 group 内区分阶段。

Tiger/diagnosis 启动脚本已经保留尾部 Hydra override，因此 `group=rkmeans` 当前可透传，但没有显式参数、允许遗漏且默认落入错误 group。此次变更跨 experiment 配置、logger 配置和 Bash 参数契约，但不涉及 Python launcher、模型或 Artifact resolver。

## Goals / Non-Goals

**Goals:**

- 让每一种量化方法的量化、推荐和诊断 run 进入同一个 W&B group。
- 保持 `sem_embeds` 作为公共语义向量数据源的独立 group。
- 让 Tiger/diagnosis 的量化方法选择成为显式且可校验的启动参数。
- 通过 `<task_name>/<timestamp>` run name 在组内清晰区分任务阶段。
- 保持现有 dry-run、notes、设备参数和尾部 Hydra override 行为。

**Non-Goals:**

- 不修改 W&B project/entity、run ID、Hydra 输出目录或 Artifact lineage。
- 不按 group 重命名 Artifact collection；不同方法的 Tiger/diagnosis Artifact 仍通过 producer run URI 和 lineage 区分。
- 不合并或重组现有 logger component 文件。
- 不从 semantic ID Artifact 自动推断量化方法。

## Decisions

1. 使用顶层 `group` 作为唯一 pipeline taxonomy 字段。

   固定生产阶段继续在 experiment 中声明 `sem_embeds`、`rkmeans`、`rvq` 或 `rqvae`。Tiger 和 diagnosis 将 `group` 改为 Hydra 必填占位符 `???`，避免保留 `tiger` 或 `tail_sid_diagnosis` 这类错误默认值。相比新增 `quantization_method` 再映射到 group，直接复用现有字段能保持配置和 W&B Config 简洁。

2. 官方 Tiger/diagnosis 脚本显式接收并验证 group。

   `tiger_train.sh`、`tiger_inference.sh` 和 `tail_sid_diagnosis.sh` 支持 `--group=value` 与 `--group value`，只接受 `rkmeans`、`rvq`、`rqvae`。缺失、空值或未知值以状态码 2 失败。脚本生成的 `group=<value>` 放在默认 Hydra 参数中，`EXTRA_ARGS` 仍最后追加，使高级用户保留显式覆盖能力。

3. 所有官方 logger 使用任务名加时间的 run name。

   10 份 logger component 的 `name` 统一设置为 `${task_name}/${now_tz:%Y-%m-%d_%H-%M-%S}`。斜杠只属于 W&B 展示名称，不用于 Hydra 输出路径；W&B `job_type` 继续来自现有静态 logger 配置。

4. 保持 Artifact 命名不变。

   W&B group 不构成 Artifact namespace。虽然不同量化方法的 Tiger 或 diagnosis run 可能发布到同名 Artifact collection，但下游协议以 `wandb://<producer-run-id>` 解析指定 run 的输出并记录精确 lineage，因此本次无需扩大为 Artifact 命名迁移。

5. 使用配置组合和脚本行为测试锁定 taxonomy。

   Hydra 测试覆盖固定 group、动态 group、必填失败和 run name 格式；脚本测试覆盖两种参数形式、三种允许值、缺失/非法值、quoted options、尾部 override 与 Bash 语法。

## Risks / Trade-offs

- [Risk] 用户为 semantic ID 选择了错误 group，W&B 分类与真实 lineage 不一致。→ 脚本限制为已知量化方法，并依赖现有 Artifact lineage 提供真实上游证据；不在本次引入远程元数据自动推断。
- [Risk] `group: ???` 会使旧的 Tiger/diagnosis 命令失败。→ 这是有意的 breaking change；错误信息将直接指向缺失 group，官方脚本提供明确 `--group` 接口。
- [Risk] `EXTRA_ARGS` 可在显式 `--group` 后再次覆盖 group。→ 保留项目既有“用户尾部 override 优先”契约，测试明确该顺序。
- [Risk] 同名 Artifact collection 仍跨量化方法共享版本。→ 当前 run-based resolver 不依赖全局 latest；若未来需要按方法浏览 Artifact，再单独设计 Artifact 命名迁移。

## Migration Plan

1. 更新 experiment 和 logger 配置并通过 Hydra compose 检查。
2. 更新三个脚本及参数测试。
3. 将现有 Tiger/diagnosis 调用补充 `--group rkmeans|rvq|rqvae`。
4. 新 run 使用新 taxonomy；历史 W&B run 不迁移。
5. 如需回滚，恢复三个 experiment 的旧默认 group、脚本接口和 logger name，不涉及数据或 checkpoint 格式回滚。

## Open Questions

无。
