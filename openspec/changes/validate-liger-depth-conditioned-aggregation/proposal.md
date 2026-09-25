## Why

偏好分散诊断表明，子树总质量能预测目标路径恢复，但其最终推荐收益被候选集合替换产生的竞争损失抵消。首次分歧发生在根层时 mass 相对 max 净损17个命中，而发生在更深层时净增48个命中，因此需要检验总质量是否应只用于粗分支确定后的细粒度路径扩展。

## What Changes

- 增加唯一的深度条件聚合控制：depth0 使用最大后代内容分数，depth1及以后使用子树总质量。
- 保持联合 checkpoint、learned alpha、beam20、合法支持、cold union 与最终内容排序不变，只改变内容前缀聚合随深度的选择。
- 输出现有候选 trace 和分散度 trace，并增加完成态配对分析，检验覆盖优势是否保留及其能否转化为 NDCG/Recall 收益。
- 冻结0次训练、1次人工 prediction；不搜索切换深度、alpha、beam、seed或subgroup。

## Capabilities

### New Capabilities

- `liger-depth-conditioned-aggregation`: 在固定联合模型推理中使用 max-root/mass-deep 内容前缀聚合，并按冻结门槛验证路径价值到推荐收益的转化。

### Modified Capabilities

无。

## Impact

影响 LIGER 内容引导 processor、联合推理控制配置、诊断 launcher、配对分析、聚焦测试和研究协议。默认行为与既有 checkpoint 保持兼容，不新增训练参数或第三方依赖；完整 prediction 仍由用户手动启动。
