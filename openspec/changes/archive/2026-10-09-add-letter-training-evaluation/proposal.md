## Why
独立骨干需要接入公共Lightning入口、checkpoint身份与共同全目录评价，才能成为可运行baseline。
## What Changes
- 新增LETTER tokenizer和推荐Lightning模块，不继承现有模型。
- 新增独立Recall/NDCG与predict评价callback；复用公共MetricEngine和writer。
## Capabilities
### New Capabilities
- `letter-training-evaluation`: 独立训练、恢复与评价。
### Modified Capabilities
无。
## Impact
新增LETTER领域模块及tokenizer tensor datamodule，不修改现有模型行为。
