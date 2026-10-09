## Why

候选并集相对mass20产生96个max-only新命中，同时因候选竞争损失81个原命中并使111个保留命中降序，最终NDCG点估计为正但区间跨0。需要在不使用已消耗testing调参的前提下，判断单参数mass来源保护能否稳定保留新准入收益并减少排序稀释。

## What Changes

- 增加一个evaluation-only候选缓存run，在共享encoder与内容投影下固定执行mass20、max20、mass30三次搜索。
- 缓存精确候选池、内容分数、mass来源标记、cold集合与目标，不上传可再生成的缓存Artifact。
- 固定按用户内容分数标准化，并扫描`beta=[0,0.1,0.25,0.5,1.0]`的mass来源加分；不训练网络或搜索其他特征。
- 使用确定性50/50 selection/audit用户划分；selection要求相邻非零beta形成正向平台，audit只评价冻结beta。
- 冻结0训练、1次evaluation缓存prediction、每用户3次beam搜索、0次新testing。

## Capabilities

### New Capabilities

- `liger-source-protection-diagnosis`: 在evaluation候选缓存上验证单参数mass来源保护是否能稳定超过原union和mass30。

### Modified Capabilities

无。

## Impact

影响LIGER evaluation-only推理模型、候选缓存schema、本地writer、离线分析、Hydra配置、人工launcher、聚焦测试和研究协议。默认模型与已完成testing结果不变，不新增依赖；完整GPU缓存仍由用户手动启动。
