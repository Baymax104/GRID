## Why

A 的完整内容条件均值与残差码质心在深层具有不同几何，但尚不清楚这种区别是否带来推荐收益。用两个限定训练对照排除“第一层已解释全部收益”和“残差初始化同样有效”，避免继续扩展复杂分支。

## What Changes

- 冻结后续testing：只使用A/匹配残差seed42–46的既有best-val checkpoint，新增手动推理包装脚本与固定判读协议；不再训练或选择候选。

- 后续候选：在固定A上仅校准深层均值范数，固定eta=0.5，先由用户手动执行seed45/46两次训练；不新增在线模块。

- 为现有 `token_content_init` 提供第一层初始化和深层残差质心初始化两种显式配置，保留 A 默认行为。
- 固定残差映射、尺度、产物身份与 checkpoint 契约，使用公共 artifact 解析及 lineage。
- 增加 Beauty seed42 两条件串行启动脚本，使用 GPU0/1、20k steps，支持 dry-run、notes 和 Hydra override。
- 补充针对性测试、预先声明的判读规则和手动启动命令。

## Capabilities

### New Capabilities

- `sid-initialization-value`: 有边界的 SID 初始化效果验证。

### Modified Capabilities

无。

## Impact

影响目录模型初始化、data 侧码本读取、组件配置、根目录脚本及测试。不新增依赖，不更改损失或在线评分，不自动运行完整实验。
