## Context

当前项目在模型训练/推理层并不依赖 TensorFlow，但 TFRecord 读取层仍用 TensorFlow 完成三个关键步骤：读取 `.tfrecord.gz`、解析 serialized example、以及把 sparse 表示转成 dense numpy。这样导致 `tensorflow-cpu` 成为运行时依赖，而它的职责实际上只是数据解码基础设施。

用户已经明确目标：保留 `tfrecord` 数据格式与上层链路契约，但完全移除 TensorFlow 运行时依赖。也就是说，新实现必须继续为现有 preprocessing、collate 和模型输入提供兼容的数据结构。

## Goals / Non-Goals

**Goals:**
- 在不依赖 TensorFlow 的前提下继续读取 `.tfrecord.gz`。
- 替换 TensorFlow example 解析与 sparse-to-dense 转换逻辑。
- 保持官方 experiment 的后续预处理 / collate / 模型输入契约兼容。
- 为后续删除 `tensorflow-cpu` 依赖创造条件。

**Non-Goals:**
- 不更换 TFRecord 数据格式。
- 不重构上层模型、collate 或 experiment 配置语义。
- 不顺带改成 parquet/arrow 等新存储格式。

## Decisions

### 1. 保留 TFRecord，重写 reader/decoder
- 决策：继续使用 `.tfrecord.gz` 作为输入格式，但把 `TFRecordIterator` 从 TensorFlow API 重写为基于非 TensorFlow reader/decoder 的实现。
- 原因：这是满足“去掉 `tensorflow-cpu`”且保持数据资产不变的最小路线。
- 备选方案：同时迁移数据格式。未采用，因为 blast radius 过大。

### 2. 兼容现有预处理链路的数据结构
- 决策：新 reader 输出的数据形态必须尽量保持与当前预处理链路兼容，例如字段值仍能被现有 `convert_fields_to_tensors`、`tokenize_text_features`、`map_sparse_id_to_embedding` 等函数消费。
- 原因：用户明确要求后续链路兼容，这要求把改动集中在读取层和必要的预处理适配层。

### 3. 同步移除 TensorFlow 类型与工具调用
- 决策：删除 `iterators.py` 和 `pre_processing.py` 中所有 `tf.*` 调用及相关类型标注，必要时改用 Python / numpy / torch 类型。
- 原因：如果代码里仍保留 `import tensorflow`，就无法真正删除 `tensorflow-cpu`。

### 4. 优先验证官方三段主 pipeline
- 决策：重点验证 `sem_embeds_inference_flat`、`rkmeans_train_flat`、`tiger_train_flat` 对应的数据链路兼容性。
- 原因：这三类实验分别覆盖 item 文本、embedding 量化、sequence semantic ID 主链路，是最关键的兼容性基线。

## Risks / Trade-offs

- [TFRecord feature 解析兼容性不足] → 需要对 bytes / float / int64 / 变长字段做逐类验证。
- [新 reader 输出 shape/dtype 与旧链路不一致] → 将验证重点放在预处理后关键字段形状与 dtype 上，而不是只看 reader 是否工作。
- [性能/并发特征变化] → 第一阶段优先保证兼容性和依赖移除，再评估是否需要进一步优化 reader 吞吐。

## Migration Plan

1. 重写 `TFRecordIterator`，去掉 TensorFlow reader 与 example parser。
2. 调整 `pre_processing.py`，移除 `tf.sparse.to_dense` 依赖。
3. 删除剩余 TensorFlow 类型标注与导入。
4. 做最小静态检查，并对三段主 pipeline 的数据链路做兼容性验证。
5. 完成验证后，再更新依赖清单删除 `tensorflow-cpu`。

## Open Questions

- 当前无阻塞性开放问题。
