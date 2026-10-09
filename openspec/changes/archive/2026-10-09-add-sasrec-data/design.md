## Context

官方 sampler 将完整用户 training 序列右移一位，截取最新 L 个监督位置，左补0；负样本均匀选自1..N，排除完整 training 序列集合。GRID 原始商品0是合法 ID；目前固定目录由 SID bundle keys 提供。

## Goals / Non-Goals

**Goals:** 保留官方监督语义、固定全目录映射和完整 training 行负采样边界，复用现有流式框架。

**Non-Goals:** 不读取 validation/test 来建立训练排除集，不实现模型、评价指标、配置或独立 runner。

## Decisions

- `ItemCatalog` 保存排序去重的非负整数 keys，模型 ID 为排序位置+1，inverse 恢复原始 key，指纹固定映射身份。使用 `load_model_output` 统一读取 `item_catalog_path`；W&B URI 必须明确 role，如复用 SID keys 则 role=semantic_id。
- 预处理可返回 None 过滤 training 长度不足2；evaluation 长度不足2报错，避免悄悄删除用户。
- training 用整条 sequence_data 的集合排除负样本，截断仅影响监督窗口，不影响排除集合。使用 torch worker RNG；有限次 rejection 后从显式合法集合采样，保证均匀性并避免满目录死循环。
- evaluation/test 固定最后商品为标量标签，无负采样，history 去除目标后左 padding。
- SASRecModelInput 只存 input_ids/output_keys；SASRecLabelData 存 target_ids/可选 negative_ids。collate 位于共享 collate.py。

## Risks / Trade-offs

- 满目录用户没有合法负样本 → 明确报错，不静默采样正例。
- 固定目录身份漂移 → 指纹在后续 checkpoint 模块校验。
- 仅从 bundle keys 读取目录 → 复用上游目录身份但不依赖 SID 算法；使用显式字段和 Artifact role。
