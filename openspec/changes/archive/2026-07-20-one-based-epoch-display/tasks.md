## 1. 自定义 progress bar 显示

- [x] 1.1 新增基于 `TQDMProgressBar` 的自定义 callback，将 epoch 文本显示为 `current_epoch + 1`
- [x] 1.2 确保该 callback 只修改显示层，不改变内部 epoch 状态或训练语义

## 2. 默认接线

- [x] 2.1 将自定义 progress bar 接入默认 callbacks 装配路径
- [x] 2.2 确认 train / validation / test / predict 默认运行都使用一致的 epoch 显示规则

## 3. 验证

- [x] 3.1 做最小静态检查，确认新增 callback 与配置语法正确
- [x] 3.2 复核控制台中 epoch 显示从 1 开始
- [x] 3.3 复核 checkpoint 恢复与内部 epoch 语义未受影响
