## MODIFIED Requirements

### Requirement: ModelOutput SHALL 直接持有 keys + predictions tensor

`ModelOutput` 作为 keyed prediction output 的字段规范层，SHALL 定义于 `src.data.components.data_models`，并直接持有 `keys` 和 `predictions` 两个属性，SHALL NOT 携带 `key_name`/`prediction_name` 等字段命名映射参数。`prediction_step` SHALL 直接构造 `ModelOutput(keys=..., predictions=...)`，使写入数据在实现中一目了然。

#### Scenario: prediction_step 直接构造 ModelOutput
- **WHEN** 任何 inference module 的 `predict_step` 产出预测结果
- **THEN** 返回值 MUST 为 `ModelOutput(keys=<业务主键>, predictions=<预测值>)`
- **THEN** `ModelOutput` MUST NOT 接受 `key_name` 或 `prediction_name` 参数
- **THEN** `ModelOutput` MUST 从 `src.data.components.data_models` 导入

#### Scenario: ModelOutput 不提供行格式转换
- **WHEN** 维护者检查 `ModelOutput` 的接口
- **THEN** `ModelOutput` MUST NOT 提供 `list_of_row_format` 属性或 `_convert_to_list` 方法

### Requirement: Writer SHALL 直接缓存和合并 tensor，不经过行格式中间层

`LocalPickleWriter` SHALL 直接缓存 `ModelOutput` 对象（而非行格式 `list[dict]`），在合并时 SHALL 直接对 `keys` 和 `predictions` 执行 `torch.cat` 产出 keyed prediction bundle。Writer SHALL NOT 持有 `prediction_key_name`/`prediction_name` 配置参数，SHALL NOT 在 `setup()` 中向 module 注入字段名。

#### Scenario: writer 缓存 ModelOutput 对象
- **WHEN** `write_on_batch_end` 接收到 `ModelOutput` 预测结果
- **THEN** writer MUST 将 `ModelOutput` 对象直接追加到缓冲区
- **THEN** writer MUST NOT 调用 `list_of_row_format` 或任何行格式转换

#### Scenario: writer 合并时直接拼接 tensor
- **WHEN** rank 0 在 `on_predict_end` 中执行合并
- **THEN** writer MUST 对所有 `ModelOutput.keys` 执行 `torch.cat` 产出 `keys` tensor
- **THEN** writer MUST 对所有 `ModelOutput.predictions` 执行 `torch.cat` 产出 `predictions` tensor
- **THEN** writer MUST 保存 `{"keys": keys, "predictions": predictions}` 到 `merged_predictions_tensor.pt`

#### Scenario: writer 不产出行格式 pkl
- **WHEN** 推理完成后检查输出目录
- **THEN** 目录中 MUST NOT 存在 `merged_predictions.pkl` 文件

#### Scenario: flush_frequency 保持行数语义
- **WHEN** `handle_batch` 判断是否需要 flush 缓冲区
- **THEN** 判断依据 MUST 为缓冲区中累计的样本数（各 `ModelOutput.keys` 长度之和）
- **THEN** `flush_frequency` MUST NOT 表示 batch 数

### Requirement: Writer SHALL NOT 依赖模块的 prediction_key_name/prediction_name 属性

`BaseBufferedWriter` SHALL NOT 在 `setup()` 中读取或设置 `pl_module.prediction_key_name` 或 `pl_module.prediction_name`。字段命名映射的职责 SHALL 完全由 `prediction_step` 承担。

#### Scenario: writer setup 不注入字段名
- **WHEN** `BaseBufferedWriter.setup()` 被调用
- **THEN** MUST NOT 访问 `pl_module.prediction_key_name` 或 `pl_module.prediction_name`
- **THEN** MUST NOT 向 `pl_module` 设置任何属性
