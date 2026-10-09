## Why

M3 固定 Beauty/seed42 的五项消融和三项机制计划缺少实际入口，正式 Full 的硬编码契约不能用临时 override 绕过。用户要求补齐入口并在 Linear 提供可验证的双卡训练、单卡推理命令，不启动正式实验。

## What Changes

- 新增独立消融模型与变体契约，复用正式计算和预算，拒绝跨变体 checkpoint。
- 新增统一 Trainer.test 的 bundle 分解、残差位置干预和前缀概率诊断，使用共享 writer。
- 新增根目录脚本及组件配置，支持 notes、dry-run 和 override 透传。
- 核对 W&B group 格式，补齐八个 issue 的命令及未产生 checkpoint 的占位说明。

## Capabilities

### New Capabilities

- `copmrec-m3-experiments`: 单数据集单 seed 消融训练、推理与机制分析入口。

### Modified Capabilities

无。正式 Full 的协议与数值行为保持。

## Impact

涉及 CoPMRec、已有 scratch 优化器验证的可扩展接口、数据产物 checkpoint 读取、共享分析 writer、新配置/脚本/测试及 Linear issue。无新依赖；不启动实验，不操作 Git 历史。

## 后续实际执行（2026-10-08）

上述不启动实验为初次准备阶段的授权边界。用户随后明确要求开始 BMX-117：训练独立 tmux 启动并报告 run ID，无训练且依赖具备的诊断执行至完成。五臂已通过有界启动核验；真实 M2 暴露 Hydra metadata 序列化故障，补充容器转换和聚焦回归，保留失败 run 后使用新 run 重跑。固定实验预算与实证解释范围保持。
