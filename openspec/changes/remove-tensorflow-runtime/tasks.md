## 1. TFRecord 读取层替换

- [x] 1.1 重写 `TFRecordIterator`，移除 TensorFlow reader 与 example parser 依赖
- [x] 1.2 确保新的 TFRecord 读取实现继续支持 `.tfrecord.gz` 输入
- [x] 1.3 保持 reader 输出结构与现有预处理链路兼容

## 2. TensorFlow 依赖清理

- [x] 2.1 从 `pre_processing.py` 中移除 `tf.sparse.to_dense` 依赖并改用非 TensorFlow 结构处理
- [x] 2.2 删除运行时代码中的 `import tensorflow` 与相关类型标注
- [x] 2.3 清理与 TensorFlow 绑定的注释/接口语义，确保运行时不再依赖 `tensorflow-cpu`

## 3. 兼容性验证

- [x] 3.1 验证 `sem_embeds_inference_flat` 的预处理后输入结构与模型契约保持兼容
- [x] 3.2 验证 `rkmeans_train_flat` 的 embedding 映射链路保持兼容
- [x] 3.3 验证 `tiger_train_flat` 的 sequence / semantic-id 链路保持兼容
- [x] 3.4 做全文搜索，确认运行时代码中已无 TensorFlow 直接依赖
