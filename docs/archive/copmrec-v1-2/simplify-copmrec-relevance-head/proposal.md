# CoPMRec v1.2：简化相关性head

## Why

校准后从零训练57hzkcol在已观测selection验证中仍未建立相对v0的推荐优势；27.5k指标也低于22.5k。等权/校准短程两臂仅支持降低排序权重，不能证明完整链路收益或归因于head。用户授权将head输入简化为完整SID的decoder末状态，作为v1.2。

## What Changes

- 新增v1.2组件、训练/推理experiment和根脚本；head仅使用d_ui，保持content加残差。
- 继承v1.1批量执行、候选规则、校准权重和全参数联合训练。
- 独立checkpoint及trace契约；保留v0/v1/v1.1行为。
- 给出从零双卡命令，测试并同步运行文件；完整训练由用户手动开始。

## Capabilities

### New Capabilities
- `copmrec-decoder-relevance`: 用完整候选SID的最终decoder状态计算相关性残差。

### Modified Capabilities

无。

## Impact

影响推荐组件、候选trace校验、配置、根脚本及聚焦测试。无新依赖，无冻结teacher，无新分析入口。
