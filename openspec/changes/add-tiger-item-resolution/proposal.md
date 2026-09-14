## Why

CGBS 首轮结果支持内容信息有用，却未支持长尾与多原型核心主张。需要直接检验：在固定 SID 目录下，学习多个前缀层的 item 解析，是否优于固定解析层和强 dense/hybrid 对照。

## What Changes

- 新增独立 MIR 推荐模块，包含目录索引、前缀条件解析器、停止/路由概率、精确目标边缘似然与预算推断。
- 提供 MIR、固定最早可行层、固定第二层、深度常数门、dense、mask CE、内容初始化、hybrid、COBRA 适配九个条件；文档明确论文方法与适配控制的区别。
- 新增单卡 inference、独立 item resolution trace，以及使用共享 writer 和现有 diagnosis 的评价流程。
- 为 WIDE 风格适配提供训练集熵标定与推理策略，不把熵标定数据混入 evaluation/testing。
- 提供两组 GPU 队列和完整三 seed 方案，统一 40k 从头训练、500 步验证、best/last 保存；正式实验由用户手动启动。
- 对共享辅助 writer 增加可选 validator 注入，默认 prefix trace 行为保持兼容。

## Capabilities

### New Capabilities

- `tiger-item-resolution`: 独立 MIR 及匹配对照、概率/身份契约、trace、训练与推断配置。
- `item-resolution-experiment-suite`: 人工启动的完整矩阵、单卡评价、标定与诊断、可审计配置和脚本。

### Modified Capabilities

无既有需求变更；共享 writer 的 validator 是向后兼容扩展。

## Impact

新增 `src/recommendation/tiger_item_resolution/`、相关 data helper、Hydra component/experiment、根启动脚本及聚焦 CPU 测试；不修改 TIGER baseline 数据流，不重训 tokenizer，不引入依赖。研究依据见 `E:/projects/research/ideas/2026-09-13-main-method-mir.md`。本提案与实现不包含自动启动完整 GPU 实验或提交 Git。
