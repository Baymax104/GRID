## Why
独立LETTER模块已就绪，需要统一Hydra入口和根脚本，并证明真实GPU训练、恢复和预测链路可运行。
## What Changes
- 四个experiment：tokenizer训练、SID导出、推荐训练、推荐推理。
- 独立组件配置和根脚本，支持dry-run、notes和额外override。
- 对齐Linear共同协议，并记录官方差异和CF准备契约。
## Capabilities
### New Capabilities
- `letter-pipeline`: 统一入口与GPU有界验证。
### Modified Capabilities
无。
## Impact
新增configs、根脚本及使用说明；仅调用公共组件及新LETTER模块。
