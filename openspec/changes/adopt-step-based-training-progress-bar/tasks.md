## 1. 实现通用 step-based Rich training progress bar

- [x] 1.1 在 `src/utils/progress_bar.py` 中新增或改造自定义 `RichProgressBar`，使 train 主进度条以 `global_step / max_steps` 为展示主轴
- [x] 1.2 保留 Rich 风格的进度条样式（条形、速度、耗时等），不退化成纯文字输出
- [x] 1.3 在通用 progress bar 中移除 `v_num`
- [x] 1.4 保持 validation / test / predict 的显示逻辑不被本次训练主轴改动破坏

## 2. 统一接入所有训练实验

- [x] 2.1 确认默认 callbacks 装配入口，并将 step-based progress bar 作为训练实验默认 progress bar 接入
- [x] 2.2 验证 `rkmeans_train`、`tiger_train`、`rqvae_train`、`rvq_train` 等训练实验配置都会自动使用该 progress bar

## 3. 优化 layer-wise step budget 日志

- [x] 3.1 更新 `src/quantization/residual_quantization.py`，显式计算 layer-wise step budget，正确处理 `max_steps` 不能整除层数的情况
- [x] 3.2 启动日志中打印完整 step budget / 边界信息，而不是仅打印整除后的 `steps_per_layer`
- [x] 3.3 层切换日志补充全局 step 信息，帮助与 train progress bar 对齐理解
- [x] 3.4 不将 `layer` / `layer_step` 注入通用 progress bar，保持该信息仅通过日志表达

## 4. 验证收尾

- [x] 4.1 最小验证：`rkmeans_train` 的 train progress bar 不再显示 `Epoch 0/-2` 风格文本
- [x] 4.2 最小验证：step-driven 训练下不再出现对 `1006/1000` 这类显示的误导性阅读
- [x] 4.3 最小验证：progress bar 中不再显示 `v_num`
- [x] 4.4 最小验证：其余训练实验配置能成功装配新的 progress bar
