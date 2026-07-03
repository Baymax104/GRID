## Why

当前项目在训练与推理主链路中并不使用 TensorFlow 作为模型框架，但 TFRecord 读取层仍直接依赖 `tensorflow-cpu` 完成文件读取、example 解析和 sparse-to-dense 转换。这使运行时依赖比实际需要更重，也增加了环境构建、平台兼容和维护成本。

现在需要在保留 `tfrecord` 数据格式的前提下，移除 TensorFlow 运行时依赖，并保证现有 experiment 的后续预处理、collate 和模型输入链路保持兼容。

## What Changes

- **BREAKING** 将 TFRecord 读取与解析实现从 TensorFlow API 替换为非 TensorFlow 方案。
- **BREAKING** 删除运行时对 `tensorflow-cpu` 的依赖，并清理代码中所有直接 `import tensorflow` 用法。
- 保留现有 `.tfrecord.gz` 输入格式与 experiment 配置，不要求上层模型/配置大幅改动。
- 保证新 reader 输出的数据结构仍兼容当前 preprocessing、collate 和模型输入契约。

## Capabilities

### New Capabilities
- `tensorflow-free-tfrecord-loading`: 在不依赖 TensorFlow 运行时的情况下读取和解析 TFRecord，并保持后续链路兼容。

### Modified Capabilities

## Impact

- 受影响代码：`src/data/loading/components/iterators.py`、`src/data/loading/components/pre_processing.py` 及必要的类型注释/文档
- 受影响依赖：最终可从依赖清单中删除 `tensorflow-cpu`
- 需要重点验证 `sem_embeds_inference_flat`、`rkmeans_train_flat`、`tiger_train_flat` 等官方实验的数据链路兼容性
