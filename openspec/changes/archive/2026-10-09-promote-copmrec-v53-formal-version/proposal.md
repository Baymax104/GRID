## Why

用户将 v5.3 确定为 CoPMRec 正式版本，要求其他开发版本退出可执行入口，并从新的训练与评价开始形成论文证据。开发 run、checkpoint 和历史完成状态不得计入正式实验。

## What Changes

- 提供统一的 `copmrec` 模型、训练和推理入口，数学行为保持 v5.3。
- 固定从头 50k 更新、双卡 global batch 256、单卡完整目录 dense 部署、权重为 1 的辅助内容视图 CE。
- 新增正式 checkpoint 契约，拒绝开发阶段 checkpoint；记录正式实验标记与来源快照。
- 归档其他 CoPMRec 开发入口、独有实现和专用测试的原始字节及 SHA256，再移除活动入口。
- 保留 v5.3 的必要内部基类、LIGER baseline 和公共上游工具。
- 正式主 baseline 为 LIGER hybrid；新增只读评价适配，保留原 gen20、cold union 和内容评分，只统一输入历史资格。dense 保留为内部对照。

## Capabilities

### New Capabilities

- `copmrec-formal-release`: 正式版本、实验入口及开发证据隔离契约。

### Modified Capabilities

无。

## Impact

影响 CoPMRec 模型装配、Hydra experiment/model/trainer、根脚本和专用测试。历史代码保存在 `docs/archive/copmrec-development-20261006/runtime/`。不启动训练、推理，不修改数据、环境或 checkpoint，不改变 LIGER 数学行为。
