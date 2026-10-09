## 1. 实现

- [x] 1.1 实现 root 包络 processor 和仅推理 model，严格复用 v3 checkpoint。
- [x] 1.2 添加独立路径 schema、观察器和校验，保持旧 trace 行为。
- [x] 1.3 添加薄配置和根脚本，固定可复制单卡命令。

## 2. 验证与交付

- [x] 2.1 验证独立概率枚举、补正一次、多用户重排、HF frontier、checkpoint 和 trace writer。
- [x] 2.2 验证 Hydra、脚本 quoting/错误输入/override/dry-run，严格验证 OpenSpec。
- [x] 2.3 Mutagen 同步并核验远端字节、真实 checkpoint 与命令；同步文档和研究状态，报告未启动正式实验。

## 3. 用户追加固定 alpha=0.813 对照

- [x] 3.1 支持仅推理权重覆盖，保持默认、checkpoint参数及全层/root混合一致，记录实际权重来源。
- [x] 3.2 验证固定概率枚举、checkpoint/内容不变、trace、Hydra透传与旧版本回归。
- [x] 3.3 更新一次单卡对照命令与研究门禁，通过严格验证、Mutagen同步和真实checkpoint只读核验后交付，不启动完整实验。
