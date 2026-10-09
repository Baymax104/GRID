## Context

v5.3 由多个内部基类实现，历史 experiment 与配置层层继承。正式入口应直接组合 LIGER 公共组件和固定的 v5.3 参数，不再依赖版本化开发配置。

## Goals / Non-Goals

目标：统一正式入口，保持数值流程，区分正式与开发证据，保留完整原始字节归档。

范围之外：启动正式实验、修改模型数学、调整损失权重、复用开发 run、修改 LIGER baseline 训练目标。

## Decisions

1. `CoPMRec` 继承必要的 v5.3 实现，仅增加正式 checkpoint 发布契约，不新增参数或损失。
2. 正式训练脚本拒绝 warm checkpoint；正式模型加载时要求严格匹配的 `copmrec_formal_release`，因此开发 v5.3 checkpoint 也不能直接进入正式推理。
3. 模型组件展平当前 defaults 后的参数；训练与推理 experiment 保持薄入口职责。
4. seed 支持非负整数，初始化和随机数使用不变；阶段计划使用 42、200、2026。
5. 单卡推理的 `--split validation` 对应 evaluation 目录，`--split testing` 对应 testing 目录，并同时设置 W&B stage tag。
6. 清理前归档源文件的原始字节，记录文件大小、SHA256 和移除清单；历史 OpenSpec change 归档后退出活动列表，不将旧路线并入正式规格。
7. LIGER pooling 需要的公共预处理以 `liger_pool_inference` 保留，与原配置字节一致。
8. 正式主 baseline 保持 LIGER hybrid：原始 20 beam 生成候选与全部 cold 商品并集，评分仍为原内容分数。只读 `HistoryExcludedHybridLiger` 通过已有排序 hook 应用共同历史资格，若原候选并集不足 10 个合法商品则明确失败，不扩大候选预算。dense 仅作为内部对照。

## Risks / Trade-offs

保留必要内部基类使结构变化最小，但类中的历史版本标识仍可见；其不再有独立实验入口。正式发布契约有意拒绝历史开发 checkpoint，因此正式推理必须等待新的从头训练。推理 checkpoint 引用和摘要仍需用户填写。

## Migration Plan

先归档与校验原字节，再移除历史入口，最后运行内存 CPU 模型测试、Hydra compose、Bash 参数检查和严格 OpenSpec 验证。代码交付不表示任何正式 run 已完成。
