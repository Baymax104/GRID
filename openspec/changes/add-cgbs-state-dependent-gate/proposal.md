## Why

CGBS 的正确内容对应关系已有证据，但原 C 中在线路由救回与误伤相抵，精确聚合未改善结果。当前每层单一全局 alpha 缺少状态适应能力，值得用最小门控检验这一限制，而不是继续扩原型或隔离辅助梯度。

## What Changes

- 新增 E 条件 `content_init_state_gate`：四层各三个零初始化参数，利用合法分支 P/Q 的归一化熵与 JS 差异调节 alpha。
- 保留内容初始化、生成混合梯度、原 C 辅助 CE、优化器和 20k 双卡预算。
- 训练输出按状态计数的门控统计，经现有 MetricCallback 记录；推理复用相同门控公式。
- 新条件拥有独立 checkpoint 契约，支持动态、base 敏感性对照及显式常量对照，记录干预来源。
- 现有训练/推理 launcher 支持 `--condition e`；完整实验由用户手动启动。

## Capabilities

### New Capabilities
- `cgbs-state-dependent-gate`: 可复现状态门控、梯度/数值契约、推理身份及单次训练交付。

### Modified Capabilities

无；原有各 arm 的行为与 checkpoint 契约保持。

## Impact

影响 CGBS 模型与领域 helper、新 gate model config、现有启动脚本和聚焦测试。无需新依赖、数据入口或实验入口。训练样本常量估计属于新训练有效后才执行的后续校准，本轮提供严格的常量注入接口与预定协议，不自动执行校准/完整实验。
