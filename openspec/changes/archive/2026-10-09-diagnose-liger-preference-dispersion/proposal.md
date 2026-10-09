## Why

现有同 checkpoint 控制显示，总质量聚合相比最大后代聚合提高目标候选覆盖，但最终 NDCG/Recall 优势未确认。当前 trace 缺少逐层内容分布与 beam 存活信息，无法判断覆盖差异是否确由“相关性分散在多个后代商品”驱动，也无法排除子树大小偏置。

## What Changes

- 增加仅诊断使用的逐层配对路径 trace，同时计算 mass 与 max，不改变实际候选决策。
- 对目标分支和最强竞争分支记录 max、log-mass、有效支持、后代数、两种规则下的分支 rank、beam 阈值与目标存活。
- 增加离线汇总，检验有效支持是否预测 mass 相对 max 的目标路径恢复，并按子树大小分层报告。
- 冻结 0 次训练、最多 2 次人工 prediction 的协议；完整 prediction 仍由用户手动启动。

## Capabilities

### New Capabilities

- `liger-preference-dispersion-diagnosis`: 在固定 checkpoint、目录、合法支持和 beam 下，生成并汇总 mass/max 的逐层配对机制证据。

### Modified Capabilities

无。

## Impact

影响 LIGER 候选 processor、候选 trace schema/writer、联合推理配置、离线证据汇总、聚焦测试和人工启动脚本。不新增训练参数或第三方依赖，不改变默认训练与推理行为。
