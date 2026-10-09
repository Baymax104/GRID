## Why

前三个模块已形成官方 SASRec 算法、数据和训练评价能力，仍需要可复现的框架配置与用户运行入口，才能用于正式基线比较。

## What Changes

- 新增分组件 train/inference Hydra 配置，官方超参数起点与 GRID 固定步数/全目录协议。
- 新增根目录脚本，支持 dry-run、notes、seed/devices/checkpoint 与额外 override。
- 添加协议/命令说明及轻量配置脚本测试，通过统一入口进行本地 dry-run/5 step/推理恢复验证。
- 本提案仅负责 pipeline 装配，不包含前三个模块算法实现；依赖它们。

## Capabilities

### New Capabilities

- `sasrec-pipeline`: SASRec 的统一入口配置、脚本与有界运行验证契约。

### Modified Capabilities

无。

## Impact

新增 configs 下独立组件、根目录脚本和中文说明。正式 logger/checkpoint/result沿用 W&B；本地逻辑验证关闭发布，不产生正式实验结论。
