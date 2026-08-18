# AGENTS.md

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

## Pipeline 契约

- 标准实验链路按职责划分为：语义向量生成、量化器训练、semantic ID 生成、推荐模型训练、推荐推理、诊断分析。
- 上游产物传给下游时必须使用明确的配置字段，例如 `embedding_path`、`semantic_id_path`、`ckpt_path`。
- 推理与中间产物读取必须遵守 model output bundle 协议：内容为 `{"keys": ..., "predictions": ...}`，不得假设是裸 tensor。
- Tail-SID diagnosis 属于分析逻辑，应通过统一入口与 `Trainer.test` 执行；不要新增离线 runner 绕过主链路。

## 数据与产物契约

- 数据路径必须通过配置表达，代码不得硬编码用户本地绝对路径。
- Item 级输入、训练、验证、测试/预测数据目录必须由 data config 明确声明，调用方不得依赖 README 中的旧目录描述。
- Hydra 输出目录必须由 `paths.output_dir` 统一传播；组件不得自行拼接不受 Hydra 管理的运行目录。
- 推理输出写入必须使用 `src/common/inference/` 中的共享协议和 writer；不要恢复旧的 `src/inference.py` 或 `src/inference/` 双入口结构。
- `LocalPickleWriter` 的合并输出必须保持单文件 model output bundle 语义，默认文件名为 `merged_predictions_tensor.pt`。

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
- `src/common/inference/` 承载跨实验复用的推理输出写入与后处理协议。
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
