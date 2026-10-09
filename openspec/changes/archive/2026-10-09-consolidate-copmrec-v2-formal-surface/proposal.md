## Why

用户固定 v2 为正式版本。现有默认入口仍指向四目标旧版本，Linear 已完成记录和 W&B 旧 run 会混淆当前正式计划，需要一次收敛并将旧证据移入文档。

## What Changes

- **BREAKING**：CoPMRec 默认训练、推理与分析入口统一为 v2；移除 native view、历史排除和迭代版本运行入口，旧运行源码进入文档归档。
- M2 九个单元回退 Todo；M3 旧计划作废并移出当前项目，建立统一模板的三项训练消融和复用产物的机制分析。
- 先保存旧结果与来源，再按生产者和依赖图删除旧 CoPMRec 主结果、消融、机制及开发 run/artifacts；保留 LIGER 等正式基线与固定上游。
- 更新研究状态、当前计划和 Linear 当前文档；不启动正式实验。

## Capabilities

### New Capabilities

- `copmrec-v2-formal-surface`: v2 单一运行面、受控消融、当前计划和历史证据清理契约。

### Modified Capabilities

无。

## Impact

CoPMRec 模型、配置、脚本、相关测试、研究文档、Linear 项目及 W&B baymaxam/GRID。LIGER 基线行为不变，无新增依赖。
