## 1. 移除配置接口

- [x] 1.1 从四个 residual quantizer 模型 YAML 中删除 `track_residuals`。
- [x] 1.2 确认所有 `configs/` 文件不再声明 `track_residuals`。

## 2. 固定 residual 输出行为

- [x] 2.1 更新 `ResidualKMeans`：删除开关接口，并无条件收集与堆叠每层 residual。
- [x] 2.2 更新 `ResidualVectorQuantization`：删除开关接口，并无条件收集与堆叠每层 residual。
- [x] 2.3 更新 `ResidualQuantizationVAE`：删除开关接口，并无条件收集与堆叠每层 residual。
- [x] 2.4 更新相关类型说明与文档字符串，声明 residual 输出始终为 tensor。

## 3. 验证

- [x] 3.1 检查实现配置和代码中不存在 `track_residuals` 的残留引用。
- [x] 3.2 对四个量化实验执行 Hydra 配置组合 smoke check，确认模型构造不需要该字段。
- [x] 3.3 对三个 quantizer 执行最小 forward smoke check，确认 residual 输出形状为 `(batch_size, n_features, n_layers)`。
