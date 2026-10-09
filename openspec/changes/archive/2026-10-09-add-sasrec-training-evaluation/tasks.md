## 1. 训练和评价

- [x] 1.1 实现 LightningModule、optimizer 注入、训练及全目录分块排名。
- [x] 1.2 实现原始商品输出、metric adapter 与 prediction callback。
- [x] 1.3 实现 checkpoint 身份校验。

## 2. 验证

- [x] 2.1 验证训练梯度、全目录/tie/chunk一致、手算metrics与checkpoint拒绝。
- [x] 2.2 通过 pytest、Ruff、strict 并更新计划。

验证：2026-09-30，算法/数据/训练评价36项测试通过，Ruff/strict通过。分块测试发现 GEMM 形状导致数学同分出现浮点差异，已使用固定 hidden 维点积归约并回归。
