## Why

已验证的算法和数据模块需要接入 GRID Lightning 链路并提供共同全目录评价；作者的 sampled evaluation 和训练中查看 test 不能用于当前比较。

## What Changes

- 新增 SASRec LightningModule：官方训练目标、配置注入 optimizer、全目录分块 TopK、原始商品 key 输出。
- 新增 item-ID metric adapter 和复用 MetricCallback 的 prediction 指标 callback。
- 添加 checkpoint 目录映射和算法/历史协议身份校验。
- 本提案只覆盖 training/evaluation 模块；依赖前两个模块，不创建运行配置或脚本。

## Capabilities

### New Capabilities

- `sasrec-training-evaluation`: 官方模型训练、全目录排名、指标及恢复身份契约。

### Modified Capabilities

无。

## Impact

新增 recommendation/sasrec/module.py、metrics.py 和 common/callbacks/sasrec_prediction_metrics.py；复用现有 MetricEngine、writer、optimizer注入协议。
