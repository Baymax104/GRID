## Context

TIGER 的 `SemanticIDEncoderDecoder` 当前在构造完成 encoder 和 decoder 后，遍历所有子模块并把 HuggingFace T5 的 `T5LayerFF` 替换为项目自定义 `T5MultiLayerFF`。官方 `tiger_train` 与 `tiger_inference` 配置都通过 `mlp_layers: 2` 启用该替换。

原生 T5 FFN 已包含 layer norm、dense activation、dropout 和 residual path。当前自定义覆盖只是加深 FFN MLP，并没有被 OpenSpec 契约、单元测试或运行入口约束为 TIGER 必需行为。

## Goals / Non-Goals

**Goals:**

- 让官方 TIGER 模型使用 HuggingFace T5 原生 FFN。
- 删除 `mlp_layers` 这个非必要结构开关和对应自定义模块。
- 保持 TIGER 的 encoder/decoder、semantic ID embedding、beam search、prefix check 和训练/推理主流程不变。

**Non-Goals:**

- 不兼容旧 `mlp_layers=2` checkpoint。
- 不引入 checkpoint key 迁移或兼容加载逻辑。
- 不重新设计 TIGER 的 T5 encoder/decoder 配置。
- 不运行完整训练或推理实验。

## Decisions

1. 删除配置开关而不是保留 `mlp_layers: null`
   - Rationale: 官方配置应只暴露当前运行路径真实需要的依赖和参数。保留空开关会继续暗示这是受支持的模型变体。
   - Alternative considered: 将默认值设为 `null` 并保留代码分支。拒绝，因为仍保留 transformers 内部类名耦合和未测试结构。

2. 删除 `T5MultiLayerFF` 文件
   - Rationale: 移除唯一调用点后，该模块没有仓库内使用者。保留未使用模块会制造后续误用面。
   - Alternative considered: 移到 experimental 目录。拒绝，因为当前项目没有该实验组件的规格、测试或运行脚本。

3. 接受 checkpoint breaking change
   - Rationale: 用户已确认不需要兼容旧 checkpoint。删除兼容逻辑能保持实现窄且清晰。
   - Alternative considered: 编写 state dict adapter。拒绝，因为会扩大迁移复杂度且没有当前运行需求。

## Risks / Trade-offs

- [Risk] 原先用自定义 FFN 训练的 checkpoint 不能直接加载 → Mitigation: 明确作为 breaking change 记录，后续使用原生 T5 FFN 重新训练或加载匹配结构 checkpoint。
- [Risk] 指标可能因模型容量下降而变化 → Mitigation: 本次仅收紧官方结构，不声明指标等价；完整训练评估留给实验阶段。
- [Risk] 外部用户通过 Hydra override 传入 `model.root.mlp_layers` → Mitigation: 官方配置与构造参数同时删除，使 unsupported override 快速失败。
