## Context

依据 ../../../../research/docs/grid-experiments/2026-09-22-cgbs-content-representation-design.md。当前content CE已是全目录监督；此次变量是商品端可学习性。

## Goals / Non-Goals

实现第一关两臂匹配训练和512用户配对评价。不实现置乱/SID控制或联合Decoder训练。

## Decisions

- 独立2层128维Transformer，4heads、FF512；128→16→128无bias adapter，U零/V随机初始化。
- 全目录CE+0.001锚定，AdamW固定lr0.0005/wd0.000001；单进程batch32/2000更新。
- 校准用户1024按用户hash选取，全部窗口排除；其余窗口按user/history去重，相同历史冲突标签拒绝。
- 不创建A网络；目录SHA、初始化SHA、数据顺序SHA、最终步数随checkpoint保存并在评价时交叉校验。
- 测试期间模型及商品bank冻结，统一writer输出配对bootstrap结果。dry-run不发布证据。

## Risks / Trade-offs

缓存不代表完整训练人群，512为开发样本；两臂可训练参数相差4096。不将本探针称稳定、创新或优于A。

## 2026-09-23 用户授权预算扩展

用户明确要求尝试10k。默认2k兼容，新增training_steps=10000贯通数据流、Trainer、checkpoint和评价校验。两臂重新从相同初始化开始；新流前64000窗口与2k一致。固定10k终点评价，保留2k失败结论，不自动晋级后两关。
