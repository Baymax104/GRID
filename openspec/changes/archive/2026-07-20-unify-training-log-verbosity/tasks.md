## 1. 训练阶段日志收敛

- [x] 1.1 修改 `src/models/quantization/residual_quantization.py`，使终端 progress bar 不显示任何 train metric
- [x] 1.2 保留 `train/quantization_loss`、`train/reconstruction_loss` 及 verbose 指标的 logger 输出，但关闭其 progress bar 显示
- [x] 1.3 搜索并检查其他训练模块中的 `prog_bar=True` 训练日志调用，统一关闭 train metric 的 progress bar

## 2. 验证与测试阶段日志收敛

- [x] 2.1 检查公共训练模块中的验证日志调用，统一关闭 val metric 的 progress bar
- [x] 2.2 检查公共训练模块中的测试日志调用，统一关闭 test metric 的 progress bar
- [x] 2.3 确保非 loss 的 val/test 指标仍保留 logger 输出

## 3. 验证

- [x] 3.1 做最小静态检查，确认相关模块语法正确
- [x] 3.2 复核训练/验证/测试阶段的终端日志不显示任何 metric
- [x] 3.3 复核所有指标仍保留在 logger 路径中
