## Why

CGBS 的内容初始化已有正向证据，但同初始化下 C（完整分支）仍未超过 A。当前证据不能区分辅助 item CE 的共享编码器梯度干扰与内容信号不足。保持 CGBS 路线，通过一个可归因的改动减少这一不确定性，避免同时调参或堆叠模块。

## What Changes

- 新增 D 条件 `content_init_full_aux_detached`，仅对辅助 item CE 的编码器输出执行 stop-gradient。
- 保留共享 query MLP、混合生成损失的完整梯度、初始化、损失权重、训练预算及推理协议。
- 为现有训练和推理脚本增加 D 选择；单次 D 训练固定使用物理 GPU 0、1，默认 B/C 行为不变。
- 验证前向和随机数状态等价、梯度路径及 checkpoint 身份，交付手动实验命令。

## Capabilities

### New Capabilities
- `cgbs-auxiliary-gradient-isolation`: 可复现、身份明确的辅助编码器梯度隔离实验条件。

### Modified Capabilities

无；现有条件语义保持不变。

## Impact

影响 CGBS 模型、现有根目录启动脚本及对应单元测试。无需新依赖、数据或训练入口。已有 C checkpoint 不能冒充 D 恢复；完整实验仍由用户手动启动。单元测试不构成梯度冲突或效果增益的实验证据。
