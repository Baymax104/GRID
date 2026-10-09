## 1. 模型与证据

- [x] 1.1 实现五个变体及完整优化器/checkpoint 契约，验证 Full 保持。
- [x] 1.2 实现三个 Trainer.test 诊断、checkpoint 输入读取及共享 epoch writer。

## 2. 入口与交付

- [x] 2.1 实现组件配置和根目录脚本，验证 dry-run/notes/quoting/override。
- [x] 2.2 聚焦 CPU/compose/脚本与 OpenSpec strict 验证。
- [x] 2.3 补齐并回读 Linear 命令、group、资源映射与来源占位。

## 3. 实际启动修复

- [x] 3.1 修复真实 Hydra metadata 的 JSON 序列化边界，补聚焦回归并同步后重新执行失败诊断；保留失败 run 和开销。
