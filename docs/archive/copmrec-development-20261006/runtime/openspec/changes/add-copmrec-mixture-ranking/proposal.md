## Why

CoPMRec 已用混合概率训练并搜索候选，但最终纯内容排序未利用完整混合判断。本变更在同一研究问题内验证混合概率能否将候选优势转为最终推荐收益。

## What Changes

- 新增 CoPMRec 专用推理子类，在同一 mass 候选与全部 cold 并集上统一 teacher forcing 打分。
- 保存内容、混合、合法生成三种排序、完整候选分数和配对收益汇总。
- 新增默认 evaluation 的推理配置与根目录 launcher，复用既有 checkpoint。
- 不修改 baseline、训练、候选搜索或名义预算，不自动运行完整实验。

## Capabilities

### New Capabilities
- `copmrec-mixture-ranking`: 同一候选集合上的混合概率终排与可复算证据。

### Modified Capabilities
无。

## Impact

新增 recommendation/data/writer 组件、配置、入口与聚焦测试；共享 launcher 仅增加 dry-run 禁用 writer 名单，无新依赖。
