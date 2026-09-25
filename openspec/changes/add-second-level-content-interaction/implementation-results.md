# 第二层内容交互实施与手动运行说明

日期：2026-09-17。实现完成；完整训练、推理、GPU性能验证与远端同步均未执行，推荐增益和新颖性未验证。

## 实现范围

- `interaction.py` 实现物品等权的加性最小二乘、连通分量 gauge、收敛检查、三类固定输入和全局 RMS 校准。每个条件只新增一个零初始化标量 `g`，实际注入系数为 `tanh(g)`。
- encoder、teacher forcing 和 CGBS 实际 beam 共用同一分支；根模型单次注册参数，原 A 的默认路径保持关闭。
- checkpoint 保存固定表、训练初始尺度、目录/投影/表 hash 与拟合记录；加载校验身份并恢复保存的尺度。构造拟合的严格 hash 可能拒绝不同数值环境产生的细微差异，应先排查环境，不绕过校验。
- 薄 model 配置与单条件根脚本保留统一入口、notes、dry-run 和最后生效的额外 Hydra override。`off` 明确选原 A 配置。

## 验证及证据边界

最终聚焦回归：**315 passed，0 skipped，41.99 秒**；五条 warning 来自依赖弃用及已有测试返回值。Ruff check 和 `openspec validate add-second-level-content-interaction --strict` 通过。复核命令：

```powershell
uv run --no-sync pytest tests/recommendation/test_second_level_interaction.py tests/test_second_level_content_script.py tests/recommendation/test_sid_initialization.py tests/recommendation/test_tiger_catalog_grounded.py tests/test_tiger_catalog_grounded_config_script.py -q
```

聚焦回归包含数学分解、零门输出/原参数梯度/随机流、非零门梯度、optimizer 单次注册、padding/SEP、decoder 因果性、真实 beam 局部分数及 inactive beam、checkpoint、六种 Hydra 装配与 Bash 参数检查。CPU 单元测试不等于实际 GPU 或多进程 DDP 验证。

真实固定 Beauty 目录的三个条件均构造成功，每个持久化 buffer 共 3,689,384 字节（约 3.52 MiB），不代表总显存或总 checkpoint 增量。原始拟合 29 轮收敛，误差约 4.98e-11；置乱重投影 28 轮收敛，误差约 7.83e-11，约 96.3% pair 地址移动。单次 CPU 构造约 1.43–2.81 秒，仅供构造检查，不能当训练吞吐结果。详见 [construction-check.json](construction-check.json)。

在线只读核验的历史 A：seed42=`5g3wpbg7`、seed43=`0fuiq3vx`，均已完成；配置快照见 [reference-a-configs.json](reference-a-configs.json)。模型与训练预算配置对齐，但历史远端源码和完整运行环境尚未独立核验，因此不能称严格同环境对照。如差异可能影响结论，先额外重跑 `off`；每个匹配 A 增加一次 20k 训练，不计入六次候选预算。

## 手动训练

所有命令在 GRID 仓库根目录执行。先确认数据位于 `data/beauty`，物理 GPU 0、1 可用；可用 `--gpus` 与 `--master-port` 调整。脚本固定两卡。实验前本地执行下面命令，并确认四个 session 都是 `Watching for changes` 且无 conflict；本轮未执行同步。

```powershell
./scripts/mutagen_sync.ps1 flush
./scripts/mutagen_sync.ps1 status
```

第一阶段只手动运行以下三个 seed42 条件，每条是独立完整训练命令；不要同时占用同一 GPU。默认不启用 dry-run，需要 smoke 时显式追加 `--dry-run`。

```bash
bash ./tiger_second_level_content_train.sh --data-dir data/beauty --condition interaction --seed 42 --gpus 0,1 --master-port 29770 --notes "level2 v1; seed42; interaction; development validation only"
bash ./tiger_second_level_content_train.sh --data-dir data/beauty --condition additive --seed 42 --gpus 0,1 --master-port 29770 --notes "level2 v1; seed42; additive control; development validation only"
bash ./tiger_second_level_content_train.sh --data-dir data/beauty --condition shuffled --seed 42 --gpus 0,1 --master-port 29770 --notes "level2 v1; seed42; shuffled control; development validation only"
```

训练种子使用 `--seed`；置乱映射种子固定为 `model.root.second_level_shuffle_seed=42`，第二阶段也保持该映射。根元数据 `interaction_pair_seed` 表示与 A 配对的训练种子。不要用额外 override 悄悄改变主要实验条件；确需偏离时应同步更新协议和元数据。

需要同环境 A 时，单独执行：

```bash
bash ./tiger_second_level_content_train.sh --data-dir data/beauty --condition off --seed 42 --gpus 0,1 --master-port 29770 --notes "level2 v1; matched A rerun; additional control budget"
```

开发门槛：interaction 相对 A 的 best-validation NDCG 至少提升 1%，同点 Recall 不下降，最后五个 validation 点 NDCG 均值不下降，并超过 additive/shuffled 的 best-validation NDCG。全部通过才将三个命令的 seed 改为 43 重复；候选训练最多六次。完整判定见 [experiment-plan.md](experiment-plan.md)。不使用已经观察过的 Beauty testing 选择版本。

## 手动单卡验证推理

训练本身每 500 步验证。若需要独立重放最佳 validation checkpoint，先从实际 run 取得 best 文件的准确 URI，再执行下列模板；占位符必须替换，不预先猜测 run 或文件名。模式须与训练相同；另两种条件相应修改 mode。`off` 使用原 `tiger_catalog_grounded_inference` model 配置。

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash ./tiger_catalog_grounded_inference.sh \
  --data-dir data/beauty --dataset beauty --group rkmeans --arm token_content_init \
  --devices '[0]' --seed 42 --data-split evaluation \
  --checkpoint-path 'wandb://baymaxam/GRID/<RUN_ID>?role=checkpoint&file=<BEST_CHECKPOINT_FILE>' \
  --semantic-id-path 'wandb://baymaxam/GRID/4vyi4o6w?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --notes 'level2 v1; best validation checkpoint replay' \
  model=tiger_second_level_content_inference model.root.second_level_mode=interaction
```

W&B 记录 run ID、输入 lineage、resolved config、实际源码及环境身份，保留 checkpoint 中的构造元数据。观察 `interaction_gate`、`interaction_alpha`、`interaction_rms_ratio`，同时按同硬件、相同稳定区间比较 step/s、峰值显存与实际 beam 指标。训练分支包含未知 pair 检查，GPU 同步成本尚未测量；若吞吐下降超过 10%，先定位成本。单元测试通过不能支持加速或效果提升的结论。
