## Context

上次排查已用真实 Beauty raw SID 拓扑复现逐节点证书开销。完整 trace 仍需保持概率界，validation 只消费推荐和 loss。用户已停止并删除旧 run。

## Goals / Non-Goals

目标：消除无用证书和 trace 写入；完整证书不再逐节点查询设备目录；保持现有推荐与预算语义。

不改变目标函数、搜索调度、Q/S、验证频率、数据或基线，不自动启动完整实验。本轮不实现 decoder KV cache，以免扩大数值和结构变更。

## Decisions

1. `generate_encoded(..., collect_trace=True)` 默认保持既有完整输出；eval 明确传 False，predict 根据 trace 开关传入。轻量路径返回空证据字典，不声称提供未计算的证书。
2. 搜索的 item 累积张量与 Top-K 路径保持一致，避免浮点求和顺序差异；只有事件写入和最终上界按需执行。
3. 目录建立不持久化的 item→各层祖先 node 索引。前沿为 antichain，每个 item 至多属于其中一个节点；批量建立节点质量表，按祖先索引汇总，再加 lower，等价于旧成员更新。证书复杂度为批量张量操作，不随前沿节点数增加 GPU 调用数。
4. 静态节点 count/depth 保留 CPU 不可变元数据，避免每批从设备下载目录；每个展开深度的 gate 一次性拷回 CPU，替代逐 node 标量同步。保留原来的 Python 排队顺序。
5. 新派生索引使用 non-persistent buffer，随设备迁移但不改变 checkpoint contract/state_dict。已有 checkpoint 保持可读。

## Risks / Trade-offs

- 概率界遗漏节点 → 与独立逐成员参考实现及完整边缘分布对比。
- trace 开关改变生成 → 9 个 arm 对比输出；解析 arm 检查上下界和预算。
- 微型单测隐藏规模问题 → 宽目录调用计数回归及已下载 Beauty 拓扑 CPU 性能探针。
- CPU 比值不能外推 GPU → 报告测试硬件边界；真实 GPU 由用户手动启动。

## Migration Plan

已终止的 run 不恢复；用户同步修复后复用原两条队列命令，从头开始。协议 `mir-v1` 保持不变，修复版本通过源码记录。正式证书仍可获得，旧 trace schema 不变。

## Open Questions

无阻塞实现的问题。实际 GPU 显存和吞吐等待用户环境验证。
