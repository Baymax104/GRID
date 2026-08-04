# AGENTS.md

## 先看这些文件
- `pyproject.toml`：当前 `uv` 环境的默认清单（Windows 最小依赖）。
- `pyproject.linux.toml`、`requirements.txt`：完整训练/推理依赖的更接近真实来源；代码里会用到这里才出现的包（如 `tfrecord`、`wandb`、`google-cloud-bigquery`）。
- `configs/experiment/*.yaml`：真正的运行入口参数来源，比 `README.md` 更可信。
- `*.sh`：当前仓库里最接近“可执行真相”的启动命令模板，已经统一成 `uv run torchrun ...`。

## 运行约定
- 一律从仓库根目录运行。统一入口 `src/main.py` 依赖 `.project-root` 做 `rootutils.setup_root(...)`。
- 优先用 `uv run`，不要照抄 `README.md` 里的 `python -m ...`。
- 多卡训练/推理的标准形态是：`uv run torchrun --nproc_per_node=<N> -m src.main experiment=<name> ...`。
- 现成脚本：
  - `sem_embeds_inference.sh`
  - `rkmeans_train.sh`
  - `rkmeans_inference.sh`
  - `rqvae_train.sh`
  - `rvq_train.sh`
  - `tiger_train.sh`

## 实际 pipeline 顺序
- 语义向量：`experiment=sem_embeds_inference`
- 训练量化器：`experiment=rkmeans_train` / `rqvae_train` / `rvq_train`
- 生成 semantic IDs：`experiment=rkmeans_inference`
- 训练生成推荐模型：`experiment=tiger_train`
- 生成推荐结果：`experiment=tiger_inference`
- 前一阶段的产物通常传给下一阶段的 `embedding_path` / `semantic_id_path`，目标文件是 `.../pickle/merged_predictions_tensor.pt`（内容是 model output bundle：`{"keys": ..., "predictions": ...}`，不是裸 tensor）。

## 数据与输出路径的坑
- `README.md` 里的数据目录描述过时；可执行配置实际读取的是：
  - item 级输入：`${data_dir}/items`
  - 训练：`${data_dir}/training`
  - 验证：`${data_dir}/evaluation`
  - 测试/预测：`${data_dir}/testing`
- Hydra 输出目录固定在：`logs/{task_name}/runs/{YYYY-MM-DD}/{HH-MM-SS}`。
- `LocalPickleWriter` 会在 `${paths.output_dir}/pickle` 下先写分片 `.pkl`，最后主进程合并成：
  - `merged_predictions_tensor.pt`（单文件 model output bundle：`{"keys": ..., "predictions": ...}`，不是裸 tensor）

## 配置行为
- 默认 `print_config=True`。若不想在启动时打印完整配置树，可在 experiment 的 `extras.print_config` 中关闭。
- 推理类 experiment 通常会在 experiment 顶层显式提供 `ckpt_path`；只有像 `sem_embeds_inference` 这种实验才会显式覆盖成 `null`。
- `src/utils/launcher.py` 是统一入口 `src/main.py` 共用的装配入口：这里实例化 datamodule、model、callbacks、loggers、trainer，并处理 checkpoint 恢复逻辑。
- `src/inference/` 只是推理输出组件包，不是运行入口；不要恢复旧的 `src/inference.py` 双入口语义。

## 代码结构（只记最影响判断的）
- `src/main.py`：统一 Hydra 入口、`extras(cfg)`，再根据 experiment 中的 `run_mode` 分发到 train / inference 链路。
- `src/data/`：自定义数据管线核心。`BaseDataModule.setup()` 先按 GPU rank 分文件，再由自定义 iterable dataloader 读 TFRecord；数据加载共享 helper 位于 `src/data/utils.py`。
- `src/inference/`：推理输出协议与落盘链路，包括 `ModelOutput`、`LocalPickleWriter`、推理结果后处理，以及 keyed prediction bundle 加载/查询工具。
- `src/embedding/`、`src/quantization/`、`src/recommendation/`：分别对应语义向量、量化器、生成推荐模型三段主 pipeline。
- `src/common/components/`：仅保留跨阶段通用组件，如 metrics、loss、scheduler；不要把推理 writer 或推理输出协议放回这里。
- `src/utils/`：保留跨域基础工具，如 launcher、logging、file I/O、Rich 输出；不要放 data 专用 helper 或 inference bundle 协议。

## 验证现实
- pytest 是当前仓库的标准单元测试入口：从仓库根目录运行 `uv run pytest`。
- 单元测试统一放在 `tests/`，只测试无需 GPU 的函数级/模块级逻辑；优先使用内存中的最小输入，不依赖真实数据目录或外部服务。
- 单元测试不得直接运行完整 experiment，不得调用 `src.main`、`torchrun`、完整 Hydra experiment，或真实 Trainer 训练/推理链路。
- 改动后优先做与你触达范围一致的最小 smoke check；对配置改动，先检查对应 `configs/experiment/*.yaml` 与脚本命令是否仍然对齐。
