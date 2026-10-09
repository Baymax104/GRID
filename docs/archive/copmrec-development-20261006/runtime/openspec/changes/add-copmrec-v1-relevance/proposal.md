## Why

用户采纳候选条件相关性组件设计，要求实现一个版本并将已有基础 CoPMRec 记为 v0。基础版本已有有效结果，但相对 LIGER dense 的最终收益未建立；v1 将推荐标签监督直接接入最终候选评分。

## What Changes

- 保留基础模型和旧 checkpoint，将其明确记为 CoPMRec v0，增加可识别的 v0 别名入口。
- 新增 v1 候选相关性 head、候选列表 CE，与原三项目标全参数联合训练；不复制或冻结推荐 teacher。
- v1 训练候选包含当前 beam、cold、content Top20 和 training 正例；推理仅使用原 beam20 与 cold，标签只用于评价。
- 新增 v1 train/inference 配置与根脚本，按最终 hybrid NDCG10 选 best，版本及评分协议进入 checkpoint 和配置。

## Capabilities

### New Capabilities

- `copmrec-versioned-relevance`: 版本边界、共同训练、候选评分及验证/启动契约。

### Modified Capabilities

无；既有 v0 和 LIGER 行为保持。

## Impact

影响 `src/recommendation/liger/`、相关 model/data/trainer/experiment 配置、根脚本、聚焦测试与交付文档。不新增依赖，不启动完整实验，不改数据或旧结果。
