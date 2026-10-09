# 删除CoPMRec v1.2，退回v1.1

## Why

用户明确撤回v1.2并要求退回v1.1；这是版本选择，不作为简化head效果的实验证据。

## What Changes

- 删除v1.2组件、配置、训练/推理脚本与专属测试。
- 撤销共享评分代码中仅为v1.2增加的覆写接口和trace支持，恢复原v1/v1.1代码。
- 撤回文档中的v1.2启动命令及当前版本声明，保留历史核验和研究记录。
- 保持v1.1校准权重0.05726763550972437、全参数训练及既有checkpoint兼容。

## Capabilities

### New Capabilities
- `copmrec-version-rollback`: 移除v1.2运行能力并恢复v1.1。

### Modified Capabilities

无。

## Impact

仅影响v1.2增加的运行文件和共享代码改动；不删除实验产物、日志、checkpoint，不启动或停止训练，不重置预算。
