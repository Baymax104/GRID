## Context

Baseline 通过 training TFRecords → SID 映射 → 连续子序列展开 → label/collate → Tiger 训练。loss 在训练和评估间共享，不能原地改为加权 loss。训练入口的 ckpt_path 用于完整恢复，不适合作为重置优化器的微调初始化。

## Goals / Non-Goals

目标：独立训练类与只读统计 helper；同起点 CE/重加权配对；原模型严格加载新 checkpoint；可复现人工启动。

非目标：改变 baseline 数据处理、改 decoder、修复历史采样规则、训练完整矩阵、宣称频率因果成立。

## Decisions

1. 长度 n 的训练行有 T=n(n-1)/2 个子序列。若 T>m（默认 m=32），现有实现有放回抽 m 次后去重，每个子序列入选概率 q=1-(1-1/T)^m；否则 q=1。1-based 位置 j 成为目标的期望次数为 (j-1)q。按训练文件单次完整遍历计算，不消耗 RNG；这是展开前文件总体的期望，不声称等于有限步数/DDP/drop_last 后的实际暴露。
2. 按层与父前缀聚合期望次数，估计条件分支概率。权重 raw=min(cap,p^(-alpha))，在每层按期望目标分布归一化，未见监督分支权重为 1；默认 alpha=0.25、cap=2、层=[2,3]。零频目标不凭空产生训练样本。归一化后加权均值为 1，权重不超过 cap；全词表 CE 的归一化范围保持 baseline，不引入合法词表 masking。
3. 新 TigerTrainingProbe 继承 Tiger，只覆盖 training_step。CE 对照调用原 loss；干预使用独立 objective。eval_step/_compute_loss/generate 原样继承。统计 lookup 使用 non-persistent buffers，不增加 checkpoint 参数键。
4. 初始化配置使用 initialization_checkpoint_path；共享 resolver 解析 URI，独立模块严格加载 state_dict，优化器/步数重置。Trainer ckpt_path 必须为 null，防止混淆恢复；保留来源 URI、统计文件摘要及 SID 指纹，写入 checkpoint metadata，输出发布仍归共享 checkpoint writer。
5. 独立薄配置继承 baseline 组件；新实验限定 2000 steps、lr=5e-5、beam=10、关闭自动 testing。保存并发布 last checkpoint 作为固定预算主比较，不按中间最优指标挑选主比较权重。允许显式 Hydra override，但必须在比较前审计 resolved config。

## Risks / Trade-offs

- 频次与实际截断训练暴露不同 → 明确期望口径，记录展开参数/文件 hash，不声称实际暴露计数。
- 严格 checkpoint 加载只能确认参数 shape，不能证明 SID 语义一致 → 手动运行前核验初始化 run 的 SID lineage。
- 同 seed 不保证跨平台或 DDP 位级重现 → 固定 worker、设备数、batch、seed；等价性测试使用内存固定 batch。
- 启动时各 rank 读取统计有 I/O 成本 → 第一版采用简单确定性只读计算，记录文件清单；不引入缓存失效复杂度。
- Warm start 阴性结果只针对当前局部探针，不否定从头训练或所有训练方法。

## Migration Plan

仅新增模块与配置，baseline 无迁移。停用新 experiment 即恢复原路径。

## Open Questions

无阻塞实现的问题。真实训练效果、统计暴露代表性与跨 seed 稳定性留给人工实验。
