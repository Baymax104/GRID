# MIR validation 耗时排查

日期：2026-09-14。范围：只读 W&B、调用链检查、CPU 临时性能探针；未修改生产代码、配置或启动脚本，未启动/终止 GPU 实验。

## 结论

已确认当前实现有重大冗余：validation 计算推荐结果后仍逐前沿节点构造每个 item 的概率上界及 Top-K 证书，随后在 `eval_step` 丢弃整个 trace。CPU 差分复现中移除这一段可保持推荐 SID、分数、搜索预算及剩余质量完全相同，搜索中位耗时从 3.505 秒降到 0.211 秒。此结果定位实现瓶颈，不是服务器 GPU 加速承诺，也不代表已排除所有远端等待问题。

## 线上只读事实

| 数据集 | run | 状态快照 | 最后训练日志 |
|---|---|---|---|
| Beauty | `f2jj3udw` | running | global_step=499，runtime=51.145 秒 |
| Sports | `cvv5weoj` | running | global_step=499，runtime=51.793 秒 |

W&B 历史每 50 步约增加 4.3–4.7 秒。两条 run 首轮 validation 指标尚未上传，系统事件已覆盖到约 6436–6437 秒，即启动后约 107 分钟。不能把所有日志空窗精确认定为 validation 时间，因为没有逐 validation batch 的远端 profiler/控制台日志。

系统抽样中物理 GPU 0–3 利用率约 24%–28%；这是节点采样，可能包含其他进程，不能作为 MIR 独占利用率或精确耗时归因。硬件元数据为 A100-SXM4-80GB。

实际 resolved config 与本地方案一致：每 500 train steps 完整验证，每卡 train batch 128、val batch 16、两卡 DDP、Q=64、S=4096、展开 batch=8、32-true。历史相同数据集 evaluation 证据分别含 22363/35598 用户；若本轮输入未变，理想均分约需 699/1113 个每卡验证 batch，文件/worker 分片可能增加尾部 batch。

## 瓶颈与证据

1. `module.py:291` 的 `eval_step` 先 teacher-forcing objective，再 `generate_encoded`，最后丢弃 trace。`trace_resolution=False` 只控制 predict 的产物输出，不关闭搜索内证据计算。
2. `search.py:115` 在 Top-K 已生成之后，对每个用户的每个剩余前缀调用 `catalog.members`，并逐节点更新 item 上界。Q 限制 decoder state，但一个 state 可以扩展大量合法子节点，前沿规模并不受 Q=64 限制。
3. `catalog.py:124` 每次成员查询都有 `int(lengths.max())`。目录 buffer 在 CUDA 时，这需要取回设备标量；外层逐节点调用还包含小 tensor 创建和索引操作。搜索中另有 gate 的 `float(...)`、route probability `.cpu()` 和 `.tolist()`，会进一步串行化。
4. `module.py:144` 每次从 BOS 重算 prefix，`use_cache=False`；搜索会重复聚合 encoder states。属于可优化成本，但本轮没有量化其在实际四层 T5/GPU 中的占比。
5. 高频全量 validation 放大以上成本；40k/500 意味着 80 次完整验证。batch=16 会增加每轮的调用次数，不能直接将 batch 比例当成 8 倍总耗时。

## 可运行的轻量复现

```powershell
uv --cache-dir tmp/uv-cache run --no-sync python -m tmp.profile_mir_validation_cpu
```

输入为已经下载的 Beauty 目录 raw SID 拓扑（12101 items），合成内容、未训练的单层小 T5、CPU 单线程；不调用完整 experiment 或真实 Trainer。探针使用内存差分临时跳过最终证书计算，不更改生产文件。

- batch=16：9 次 decoder 调用；32782 次 members 查询，其中 32774 次是单节点查询。
- cProfile：4.542 秒搜索；上一轮同条件采样中 members 实现累计约 2.27 秒，另有外层逐节点 Python/索引开销。
- 无 profiler、交替 3 次：原始 3.014/3.505/3.855 秒；跳过证书 0.277/0.211/0.194 秒；中位比值 16.59。
- 三次均严格验证 SID、分数、states/items budget、remaining/resolved mass 张量完全相同。
- 不据此宣称真实训练加速 16.59 倍，也不将 CPU objective/search 比值作为服务器训练/验证比值。

## 排查过的替代解释

- validation dataset 的 `is_for_training=False`，遍历完成后退出；没有沿用训练的无限数据循环。
- 验证集没有训练侧 causal sequence expansion；不能解释为验证也膨胀了 32 倍。
- 指标在 epoch end 计算/同步；W&B checkpoint writer 的发布发生于 `on_train_end`，不是每个 validation batch 上传。
- 本地没有 CUDA，无法完成远端 CUDA profiler，尚不能严格排除分片不均、DDP 等待或其他进程竞争。

## 建议的下一步修复范围

优先在新提案中定义保持实验语义的加速：validation 仅产生评估所需结果；完整 inference trace 按需启用，概率上界向量化；减少逐节点 CPU/GPU 同步。先保持 Q/S、Top-K、验证集、500 步选择频率与训练目标不变，用输出等价检查和真实 GPU 单 batch 计时验收。

之后再根据实际显存与吞吐调整 validation batch。降低验证频率、减少验证用户、缩小搜索预算会改变 checkpoint 选择或评价协议，不能作为无影响的性能修复混入现有 `mir-v1`。

不建议按现实现继续推进整个 54-run 队列。当前异常是工程性能问题，不能用来否定 MIR 研究假设；已有 CPU 正确性测试未覆盖真实目录规模与 CUDA 同步成本，这是上一轮验收的缺口。

原始快照和探针输出位于 `tmp/mir_validation_audit_2026-09-14/`：`runs.json`、两个 history 文件、`system.json`、`cpu-profile-summary.json`、`cpu-profile-b16.txt`、`differential.json`。
