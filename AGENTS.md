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
  - `tiger_train.sh`

## 实际 pipeline 顺序
- 语义向量：`experiment=sem_embeds_inference`
- 训练量化器：`experiment=rkmeans_train`
- 生成 semantic IDs：`experiment=rkmeans_inference`
- 训练生成推荐模型：`experiment=tiger_train`
- 生成推荐结果：`experiment=tiger_inference`
- 前一阶段的产物通常传给下一阶段的 `embedding_path` / `semantic_id_path`，目标文件是 `.../pickle/merged_predictions_tensor.pt`（内容是 keyed prediction bundle：`{"keys": ..., "predictions": ...}`，不是裸 tensor）。

## 数据与输出路径的坑
- `README.md` 里的数据目录描述过时；可执行配置实际读取的是：
  - item 级输入：`${data_dir}/items`
  - 训练：`${data_dir}/training`
  - 验证：`${data_dir}/evaluation`
  - 测试/预测：`${data_dir}/testing`
- Hydra 输出目录固定在：`logs/{task_name}/runs/{YYYY-MM-DD}/{HH-MM-SS}`。
- `LocalPickleWriter` 会在 `${paths.output_dir}/pickle` 下先写分片 `.pkl`，最后主进程合并成：
  - `merged_predictions.pkl`
  - `merged_predictions_tensor.pt`（单文件 keyed bundle）

## 配置行为
- 默认 `print_config=True`。若不想在启动时打印完整配置树，可在 experiment 的 `extras.print_config` 中关闭。
- 推理类 experiment 通常会在 experiment 顶层显式提供 `ckpt_path`；只有像 `sem_embeds_inference` 这种实验才会显式覆盖成 `null`。
- `src/utils/launcher_utils.py` 是统一入口 `src/main.py` 共用的装配入口：这里实例化 datamodule、model、callbacks、loggers、trainer，并处理 checkpoint 恢复逻辑。

## 代码结构（只记最影响判断的）
- `src/main.py`：统一 Hydra 入口、`extras(cfg)`，再根据 experiment 中的 `run_mode` 分发到 train / inference 链路。
- `src/data/`：自定义数据管线核心。`SequenceDataModule.setup()` 先按 GPU rank 分文件，再由自定义 iterable dataloader 读 TFRecord。
- `src/models/embedding/`、`src/models/quantization/`、`src/models/recommendation/`：分别对应三段主 pipeline。

## 验证现实
- 仓库里没有已检查入库的 `tests/`，也没发现 repo-local lint / typecheck 配置；不要假设存在 `pytest`, `ruff`, `mypy` 的标准入口。
- 改动后优先做与你触达范围一致的最小 smoke check；对配置改动，先检查对应 `configs/experiment/*.yaml` 与脚本命令是否仍然对齐。
