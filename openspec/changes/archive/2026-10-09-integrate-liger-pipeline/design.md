## Context

参考 facebookresearch/liger commit b6ccc37af5ee623ddc1d1ead3490c31aaeaf4524，训练与预测遵守 GRID 的统一入口和 bundle 协议。

## Goals / Non-Goals

依赖前两个提案；新增薄 experiment/component 配置、训练与推理 Bash 入口、文档及 CPU 测试；统一入口执行本地 GPU dry run。

目标为忠实的 LIGER 方法在 GRID 数据协议中的适配，不声称复现论文数值，不改变既有 TIGER/CGBS 行为。

## Decisions

- 使用独立模块而非 CGBS 子类，避免继承混合概率与额外约束。
- LIGER 独立切分最后商品标签，不继承 TIGER 的目标占位 token；随后复用标准历史截断和 padding。
- 真实 Windows dry run 发现公共 TFRecordReader 将 file://E:/ 路径误判为 UNC；只修复盘符 authority 识别并回归标准 file URI/UNC 行为。
- 保留官方共享投影、全词表 T5 CE、无 EOS 目标的固定长度 SID 生成；GRID 的完整 SID 含去重位。
- 原始内容维度从 bundle 获取，不做 PCA；训练商品集合仅从 training 数据取得，冷启动为目录减训练商品集合。
- 明确记录数据划分、历史窗口与官方差异，官方代码的非法候选 argmax 映射和重复候选以安全去重处理。

## Risks / Trade-offs

- [GRID 数据并非官方数据协议] → 文档显式标为 GRID-adapted baseline。
- [重复 SID 使内容 lookup 不唯一] → fail closed，要求完整去重 SID。
- [全目录 CE 成本] → 可配置目录投影分块；先做有界 dry run，不自动全量训练。
- [checkpoint 输入错配] → 保存目录身份与训练集合并严格核验。
