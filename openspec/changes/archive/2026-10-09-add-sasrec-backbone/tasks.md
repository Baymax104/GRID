## 1. 官方骨干

- [x] 1.1 实现官方 attention、FFN、embedding 与位置序列计算。
- [x] 1.2 实现共享打分、逐位置 BCE、embedding L2 和输入校验。

## 2. 验证

- [x] 2.1 完成固定权重 NumPy 官方公式参考、因果性、损失和梯度测试。
- [x] 2.2 通过聚焦 pytest、Ruff 与 OpenSpec strict 并记录证据。

验证：2026-09-30，`uv run --no-sync pytest tests/recommendation/test_sasrec_backbone.py -q` 12 passed；Ruff 和 strict 通过。CPU 内存测试，不是实验结果。
