## 1. 量化脚本参数契约

- [x] 1.1 为 RKMeans、RVQ、RQVAE 三个训练脚本增加必填 `--embedding-path` 的两种解析形式、缺失值校验和动态 Hydra override
- [x] 1.2 为 RKMeans、RVQ、RQVAE 三个推理脚本增加相同的 embedding 参数契约，并保持 `--ckpt-path` 行为
- [x] 1.3 确认六个脚本不再引用固定的 `wandb://vb8es5ow`，且尾部额外 Hydra override 顺序保持不变

## 2. 自动化验证

- [x] 2.1 增加六个量化脚本的参数化测试，覆盖 equals/separated syntax、本地/W&B 路径、缺失值、既有选项和尾部 override
- [x] 2.2 对六个脚本运行 Bash 语法检查并执行聚焦 pytest
- [x] 2.3 运行相关回归测试与 Ruff 检查，确认未影响现有 W&B group 和 diagnosis 脚本契约

## 3. OpenSpec 校验

- [x] 3.1 使用 strict 模式校验 `require-explicit-quantization-embedding-input` change
