## Context

当前 `SequenceDataset` 在训练模式下是无限迭代的 `IterableDataset`：当底层数据读完后会重新 `setup()` 并继续迭代。因此训练不会因为“跑完一个 epoch”而自然结束，而是依赖 Trainer 的 `max_steps` 停止。这与多个 trainer / model 配置一致：

- 训练配置普遍使用 `max_steps`
- scheduler steps 也绑定 `trainer.root.max_steps`
- validation 常通过 `val_check_interval` 按 step 触发

但当前 progress bar 仍显示 epoch 文本，且在 unbounded dataset 场景里出现 `Epoch 0/-2` 这类内部语义泄露。对 step-driven 训练来说，这比 0-based/1-based epoch 更本质地误导用户。

此外，`ResidualQuantization` 在 `train_layer_wise=True` 时用 `total_steps // n_layers` 估算 `steps_per_layer`，日志打印如“each for 333 steps”。当 `max_steps` 不能被层数整除时，该日志缺少对余数分配的解释，容易与 progress bar 的总 step 显示形成阅读冲突。

用户进一步明确：训练主进度条希望采用 **RichProgressBar 风格**，并去掉默认的 `v_num`；但 `layer` / `layer_step` 不应进入通用 progress bar，因为这会污染其他训练实验的通用显示语义。

## Goals / Non-Goals

**Goals:**
- 让训练主 progress bar 明确表达“当前第几 step / 总 step 预算”
- 将训练主进度条切换为 Rich 风格，同时保留进度条而不是退化成纯文本输出
- 去掉默认进度条中的 `v_num`
- 统一适用于项目中的所有训练实验
- 让 layer-wise 训练日志更准确地表达 step budget

**Non-Goals:**
- 不把整个训练体系从 step-driven 改成 epoch-driven
- 不改变 validation / test / predict 的主显示语义，除非实现中顺带需要微调
- 不改变 trainer 内部停止条件、optimizer step、checkpoint 恢复或 scheduler 语义
- 不在本次引入新的第三方进度条依赖（Rich 已是 Lightning 支持路径）

## Decisions

### D1: 训练主进度条改为 step-based，并采用 RichProgressBar 风格
- **选择**：基于 Lightning 的 `RichProgressBar` 做定制，使 train bar 围绕 `global_step / max_steps` 展示。
- **理由**：用户明确要求更美观的 Rich 风格，同时保留进度条可视化表达。Rich 更适合做统一列布局和去噪展示。
- **备选**：继续使用 `TQDMProgressBar`——否决，虽可实现 step-based，但样式上限有限；纯文本日志显示 `Step xx/yy`——否决，丢失 bar 的可读性。

### D2: 全项目训练实验统一走默认 step-based progress bar
- **选择**：通过统一默认 callbacks 装配路径接入 step-based progress bar，而不是在单个 experiment callback 文件里逐个配置。
- **理由**：用户要求“适用于本项目中的所有训练实验”；统一默认入口可避免配置漂移。

### D3: 通用 progress bar 去掉 `v_num`，且不承载 layer 特有信息
- **选择**：在自定义 progress bar 中移除 `v_num`，并明确不将 `layer` / `layer_step` 等实验特有信息纳入通用 bar。
- **理由**：`v_num` 对日常阅读帮助有限；`layer` / `layer_step` 仅适用于少数实验，放入通用进度条会损害可复用性。

### D4: validation/test/predict 先保留原显示风格
- **选择**：本次只定制 train 主进度条；其余阶段保持现状。
- **理由**：当前用户问题聚焦训练；先把 train 的主语义纠正，避免一次性扩散改动面。

### D5: `ResidualQuantization` 明确 layer-wise step budget，而不是只打印整除后的 `steps_per_layer`
- **选择**：在训练开始时显式计算每层 step budget（处理余数分配），并在日志中输出完整预算与层切换边界。
- **理由**：这样可以解释为什么 `max_steps=1000` 时不应被阅读为简单的 `333 * 3`。这类 layer 特有语义保留在日志里，而不是塞进通用 progress bar。
- **备选**：仅保留现有 `total_steps // n_layers` 日志——否决，仍有阅读歧义。

## Risks / Trade-offs

- **[风险] 直接改 Rich train progress bar 的 `n/total` 绑定方式，可能与 Lightning 内部更新时机不完全一致** → 缓解：在自定义 Rich progress callback 中显式绑定 `global_step / max_steps`，并针对 stop 边界做最小验证
- **[风险] 默认 callbacks 接入路径当前不够显式，可能有训练实验未自动装配 progress bar** → 缓解：实现前先确认默认 callback 装配入口，并在验证阶段覆盖多个训练实验配置
- **[权衡] validation/test/predict 仍可能显示 epoch 相关文本** → 可接受，本次目标是先解决训练主轴语义错位

## Migration Plan

1. 新增或替换为 step-based Rich train progress bar callback
2. 通过默认训练 callbacks 装配路径统一接入
3. 在 progress bar 中移除 `v_num`
4. 更新 `ResidualQuantization` 的 layer-wise step budget 日志
5. 用 `rkmeans_train` 作为主验证样例，确认不再出现 `Epoch 0/-2` 且 step 展示更清晰
