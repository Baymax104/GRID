## Why

当前项目的训练实验实际采用的是 **step-driven** 训练语义：`SequenceDataset` 在训练模式下无限迭代，Trainer 主要通过 `max_steps`、`val_check_interval`、scheduler step 数等配置来控制训练预算与节奏，而不是依赖“完整跑完一轮数据集”的 epoch 语义。

但控制台 progress bar 仍沿用 epoch 视角，导致在训练日志中出现诸如 `Epoch 0/-2`、`1006/1000` 之类的显示。这对用户存在两类误导：

1. 在 step-driven 训练里继续突出 epoch，会让用户误以为 epoch 是当前训练主轴；
2. 在 `rkmeans_train` 这类 layer-wise step 训练中，进度条与日志没有准确表达 layer step budget 与全局 step 的关系，阅读时很难判断“当前跑到第几步”“当前在第几层”“为什么总 step 看起来超过 max_steps”。

因此需要将训练进度条统一改为 **step-based**，并采用 **RichProgressBar 风格**，同时保留进度条样式，而不是退化成纯文字日志。

## What Changes

- **BREAKING（显示语义）** 将默认训练 progress bar 从 epoch-based 展示改为 step-based 展示：训练主进度条以 `trainer.global_step / trainer.max_steps` 为主轴
- 将训练主进度条切换为 **RichProgressBar 风格**，保留条形、速度、耗时等可视化信息
- 去掉默认进度条中的 `v_num` 显示，减少噪声
- 统一接入到本项目所有训练实验，而非只对 `rkmeans_train` 单独生效
- 对 `rkmeans_train` 的 layer-wise 训练日志补充更准确的 step budget 表达，避免 `1000` step 被日志简化成 `333 * 3` 的歧义；该类特有信息继续通过日志表达，不进入通用 progress bar
- validation / test / predict 的显示先保持现有风格，本次只优化 train 主进度条

## Capabilities

### New Capabilities
- `step-based-training-progress-bar`: 规定训练主进度条必须以 step 为主轴展示，并采用 Rich 风格的统一进度条样式

### Modified Capabilities
- `layer-wise-step-budget-logging`: 对按 layer 分配训练 step 的实验，日志必须准确表达每层 step budget 与全局 step 边界

## Impact

- 受影响代码：`src/utils/progress_bar.py`、默认 callbacks 装配路径、必要时 `src/utils/launcher_utils.py`
- 受影响训练逻辑展示：`src/quantization/residual_quantization.py` 的 layer-wise step budget 日志
- 受影响范围：所有训练实验（如 `rkmeans_train`、`tiger_train`、`rqvae_train`、`rvq_train`）
- 不改变训练本身的 step 预算、优化器步数、checkpoint 恢复语义或 scheduler 语义，只改变显示层与日志表达
