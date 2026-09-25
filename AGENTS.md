# AGENTS.md

## 研究目标与决策约束

- 正式研究范围为“利用商品内容的生成式推荐方法”，以 LIGER 等同类方法作为主要 baseline。TIGER 保留为基础架构和基础对照，A 保留为内部强对照；不再仅以改进 TIGER/A 定义论文问题。具体比较名单与协议以研究状态和当前计划为准。
- 归档边界遵守 `../research/research-state.yaml` 的 `historical_evidence_policy`：过时或范围外实验仅供文档追溯，不再作为当前路线选择、预算、晋级或否决的依据；当前决策从明确保留的相关证据和新实验出发。技术依赖保留不代表恢复其历史研究结论。
- 项目的最终目标是产出可投稿的 DASFAA 论文；优先形成清晰的叙事角度、完整的论证与相对合理基线的可验证效果，贡献在整篇论文层面评价，不要求每个实验或组件独立新颖。
- 论文路线收敛为“一个有边界的问题 + 一个针对性机制 + 一组与主张相称的实验”。保留匹配对照和机制验证，允许多个因素共同作用，不要求单一根因解释全部失败或干预改善所有样本；分别判断效果、新颖性和机制支持。
- 实验须围绕同一个已界定问题持续收敛不确定性，沿用阶段问题、评价标准和累计预算；每次实验明确缩小哪项已有不确定性，以及正向、负向和不确定结果如何改变决策。失败不自动扩展新问题、模块或预算。
- 提出或实施新方案前，说明问题证据、相关工作定位、可检验预测及最小必要验证；尚未定型的贡献表述可在写作中完善。证据要求与主张相称，允许原因保持未知；不以穷尽反例或排除所有替代解释作为行动或停止前提。
- 深度研究与论文探索遵守 `../research/AGENTS.md` 的“论文探索与叙事定位”：相似方法或思路用于定位和选择对照，不自动判为高新颖性风险；不要求全新问题或全新组件，不因相似性强迫实验转向。实验验证效果与机制，写作组织叙事与完整贡献，表述始终受真实证据约束。
- 若证据否定瓶颈或机制，应停止或收缩该路线、保留否证结果并修订主张；不通过反复调参、挑选有利结果或更换方法名称维持原叙事。
- 研究状态及具体暂停决定以 `../research/research-state.yaml` 和 `../research/ideas/current-plan.md` 为准；上述目标不自动授权启动完整实验，完整实验仍由用户手动开始。

- 规划实验、解释结果、讨论下一方向或新增预算时，遵守 `../research/AGENTS.md` 的“减少不确定性的执行约束”和“路线判断门禁”。仅保留会改变决策的检查；阶段预算到达后作出取舍，不将“最多三个实验”逐轮重置。切换问题须先总结原问题并明确路线变更及正面依据；建议不等于恢复暂停路线。

## 上下文来源规范

- 修改前必须优先读取与任务相关的可执行配置、脚本和测试；不要只依赖 README 或历史说明判断行为。
- 依赖判断必须以项目清单文件和实际 import 为准；遇到平台差异时先确认运行环境再调整。
- Experiment 行为必须以 `configs/experiment/*.yaml` 及其 defaults 组合后的组件配置为准。
- 根目录启动脚本是用户运行实验的主要入口；修改配置行为时必须同步检查脚本参数是否仍然对齐。

## 运行入口规范

- 所有命令必须从仓库根目录运行。
- Python 入口统一使用 `src/main.py`，通过 Hydra `experiment=<name>` 选择实验；不要新增绕过统一入口的训练、推理或分析入口。
- 优先使用 `uv run`；不要使用裸 `python`、`pip` 或直接照搬过时 README 命令。
- 多卡训练或推理应使用 `uv run torchrun --nproc_per_node=<N> -m src.main experiment=<name> ...` 形态。
- 训练与 diagnosis 脚本必须支持脚本级 `--dry-run`，并将其交给统一入口的 dry-run 逻辑处理。
- 训练与 diagnosis 脚本必须支持 `--notes="..."` 和 `--notes "..."`，并将非空 notes 透传到 `logger.wandb.notes`。
- 训练与 diagnosis 脚本必须保留额外 Hydra override 的透传能力；默认参数之后的用户 override 应具有覆盖能力。
- 训练脚本不得默认启用 dry run；需要 smoke run 时必须显式传入 `--dry-run`。

## 代码同步规范

- 本地 Git 工作区是代码的唯一可信源；远端 `node1:/data3/weizhenyu/projects/GRID` 的 Git 状态不可信，不得用它判断代码版本、同步方向或完成状态，也不得通过同步修复远端 `.git/`。
- 远端是运行环境和重资产的可信源。`.git/`、`data/`、`pretrained_models/`、`.venv/`、`venv/`、`logs/`、`wandb/`、checkpoint 与实验产物必须留在同步边界之外。
- Mutagen 只管理 `src/`、`configs/`、根目录 `*.sh`、`*.ps1`、`pyproject.toml` 和 `uv.lock`。受管范围使用 `one-way-replica`：本地创建、修改和删除均传播到远端，远端受管目录中的额外代码和缓存可被清理，任何远端内容都不得反向传播。
- 同步配置以 `mutagen.yml` 为准，生命周期操作统一从仓库根目录使用 `./mutagen_sync.ps1`；不得用全仓库 replica、rsync 或临时 scp 替代该入口。同步依赖清单不授权自动修改远端虚拟环境。
- 首次或终止后重建 session 时，依次执行 `./mutagen_sync.ps1 start`、`status`、`resume`。`start` 必须只创建 paused session；在 `status` 确认三个端点、`One Way Replica` 和 ignore 边界后才能 `resume`。
- 从包含 `grid-scripts` 的旧四 session 配置迁移到当前三 session 配置时，先用旧会话仍可识别的项目入口执行 `stop`，再按 `start`、`status`、`resume` 重建；不得让已移除的 `grid-scripts` 会话继续后台运行。
- 将代码交给远端实验前必须执行 `./mutagen_sync.ps1 flush`。只有命令成功、三个 session 均为 `Watching for changes` 且无 conflict，才能报告同步完成；用 `status` 查看快照，用 `monitor` 持续观察，用 `pause`/`resume` 暂停或恢复，仅在明确需要终止同步时使用 `stop`。

## Pipeline 契约

- 标准实验链路按职责划分为：语义向量生成、量化器训练、semantic ID 生成、推荐模型训练、推荐推理、诊断分析。
- 上游产物传给下游时必须使用明确的配置字段，例如 `embedding_path`、`semantic_id_path`、`ckpt_path`。
- 推理与中间产物读取必须遵守 model output bundle 协议：内容为 `{"keys": ..., "predictions": ...}`，不得假设是裸 tensor。
- Tail-SID diagnosis 属于分析逻辑，应通过统一入口与 `Trainer.test` 执行；不要新增离线 runner 绕过主链路。

## 数据与产物契约

- 特征缓存、候选缓存等可再生成的中间缓存默认只保存在运行机器本地（当前为 node1），不上传 W&B；缓存 writer 和配置默认关闭发布，下游优先使用本地 manifest 路径。保留 manifest、分片哈希及来源指纹，W&B 继续记录运行配置与指标；checkpoint 和最终评价结果沿用各自产物发布规则。仅在用户明确要求时上传缓存，不自动删除已上传的历史产物。

- 数据路径必须通过配置表达，代码不得硬编码用户本地绝对路径。
- Item 级输入、训练、验证、测试/预测数据目录必须由 data config 明确声明，调用方不得依赖 README 中的旧目录描述。
- Hydra 输出目录必须由 `paths.output_dir` 统一传播；组件不得自行拼接不受 Hydra 管理的运行目录。
- 推理输出写入必须使用 `src/common/writers/` 中的共享 writer；不要恢复旧的 `src/inference.py` 或 `src/inference/` 双入口结构。
- `LocalPickleWriter` 的合并输出必须保持单文件 model output bundle 语义，默认文件名为 `merged_predictions_tensor.pt`。
- Artifact/bundle 读取入口归 `src/data/components/artifacts.py`，包括 `load_model_output`、`load_semantic_id_tensor` 与字段级引用解析；底层 `wandb://` URI 解析和 W&B Artifact 下载 helper 归 `src/utils/wandb.py`；不要在 launcher、model 或 writer 中实现 W&B Artifact 下载逻辑。
- 输出写入与发布归 `src/common/writers/`；`LocalPickleWriter` 和 W&B Artifact writer 必须平级且低耦合，使用本地 writer 不应触发 W&B。
- W&B Artifact lineage 记录归 `src/common/callbacks/WandbArtifactLineageCallback`；它是 callback，不是 logger 或 writer，只负责对已解析的上游 Artifact 调用 `use_artifact`。

## 配置与日志规范

- Experiment 文件应保持薄入口职责：声明 defaults、运行元信息、必要人工输入和高层装配关系；组件参数应放在对应 component config 中。
- `src/utils/launcher.py` 是统一装配入口，负责实例化 datamodule、model、callbacks、loggers、trainer，并处理 checkpoint 恢复与 dry-run override。
- Logger 配置属于 logger component；W&B 的 `project`、`group`、`notes` 等参数应通过 `configs/logger/*.yaml` 或 Hydra override 配置。
- `pipeline_launcher` 必须通过 Lightning `logger.log_hyperparams(...)` 将 resolved Hydra 配置记录到所有 logger。
- 若实验配置启用了 W&B logger，实验结果默认应从对应 W&B run 中获取，并以 W&B 中的 metrics、summary、config 和 notes 作为对比与分析依据。
- 结构化实验条件必须放入 Hydra config / override，便于 W&B Config 后续筛选；自然语言实验意图和对比说明放入 W&B notes。
- Metric 历史曲线与 summary scalar 的选择属于 `MetricCallback.logging_modes`，不得在具体 metric 内硬编码 logger 行为。

## 模块边界规范

- `src/data/` 负责数据读取、预处理装配、dataset/datamodule 和 data 专用 helper。
- `src/data/components/data_models.py` 负责运行时 batch/model output 数据容器。
- `src/data/utils.py` 可承载数据加载共享 helper 和 keyed prediction bundle 查询工具。
- `src/embedding/`、`src/quantization/`、`src/recommendation/` 分别承载语义向量、量化器、推荐模型的领域逻辑。
- `src/common/writers/` 承载跨实验复用的输出 writer 与 writer 后处理协议。
- `src/common/loss/`、`src/common/scheduler/` 承载跨阶段复用的 loss 与 scheduler。
- `src/utils/` 只放跨域基础设施，例如 launcher、logging、file I/O、Rich 输出；不要放 data 专用 helper 或 inference bundle 协议。

## 验证规范

- pytest 是标准单元测试入口，必须从仓库根目录通过 `uv run pytest` 执行。
- 单元测试应放在 `tests/`，优先使用内存中的最小输入；不要依赖真实数据目录、外部服务或 GPU。
- 单元测试不得直接运行完整 experiment、`src.main`、`torchrun`、完整 Hydra experiment 或真实 Trainer 训练/推理链路。
- 配置改动必须至少做 Hydra compose 或等价的轻量验证。
- 脚本改动必须做 shell 语法检查；涉及参数解析时必须验证 quoting、空值、错误输入和额外 override 透传。
- OpenSpec 覆盖的中大型变更必须在完成后运行对应 `openspec validate <change> --strict`。
- 验证范围应与改动风险匹配；不要用无关的大范围重跑替代聚焦测试。
