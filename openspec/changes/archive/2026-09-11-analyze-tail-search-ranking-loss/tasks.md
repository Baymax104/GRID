## 1. 前置依赖与输入契约

- [x] 1.1 确认前置 prefix instrumentation 实现、schema 与聚焦测试可用，保留未提交变更和旧 pilot No-Go。
- [x] 1.2 为 diagnosis config/data/batch 增加可选 widened recommendation 输入与两个默认关闭的分析开关；启用 paired analysis 时验证四份输入齐全。
- [x] 1.3 通过共享 resolver 保留原引用、resolved path、source run 和 Artifact identity；实现 key/label/split/checkpoint/SID/width 配对审计以及本地来源未验证状态。
- [x] 1.4 实现 recommendation target rank/membership 与 trace 最终状态交叉验证，覆盖非法 SID、重复/缺失/额外 key、lineage 冲突和不一致标签的内存测试。
- [x] 1.5 为根脚本增加 widened recommendation flag，保持 data-dir 必填、seed 42、notes、dry-run 与末尾 override 契约；同步组件配置。

## 2. 搜索—排序分解

- [x] 2.1 实现六状态逐用户分类，测试固定命中保留、demotion、候选丢失、新增 Top10、候选内低排名和完全不可达。
- [x] 2.2 输出 All/Head/Mid/Tail/Tail-Cold 的 fixed/widened Hit@10、NDCG@10、候选可达率及全部计数；覆盖零支持、rank 10/11、重复候选最早 rank 和组汇总守恒。
- [x] 2.3 输出新增、丢失、净增命中和候选内 oracle ceiling/headroom，验证 oracle 不低于对应实际 Hit@10，合并 Tail 使用加权分母。

## 3. 逐层退出与频次描述

- [x] 3.1 输出累计 survival、at-risk support、条件退出率和首次退出分布；验证完整存活 sentinel、零 at-risk 和 parent 先前退出的处理。
- [x] 3.2 输出首次失败时 teacher rank、parent beam rank、cutoff margin 的描述与有效/缺失数量，不把 teacher rank 当全局 beam rank。
- [x] 3.3 输出训练频次分箱的逐层描述，将合法 continuation 数明确标记为 catalog branching proxy；保留 legacy 相关指标的解释边界。

## 4. 静态风险标准化与区间

- [x] 4.1 聚合同 item 用户，实现默认 5 箱、重复边界合并、每组每箱至少 20 items、共同权重；测试输入顺序不变性与 item/user 权重区别。
- [x] 4.2 输出共同支持、保留比例、raw-damage SMD 与质量状态，覆盖无重叠、低保留、平衡失败和零方差情况。
- [x] 4.3 实现逐层 item-macro 标准化 survival 差距与独立标识的条件退出估计；测试已知构造值与零分母处理。
- [x] 4.4 在 prefix cluster 统计量上实现同估计量 bootstrap，保留箱边界并重算权重；测试 cluster multiplicity、固定 seed、有效重复不足与原始/标准化 CI 区分。
- [x] 4.5 固定主配置和 3/5/10 箱敏感性配置，记录 estimator version 和质量默认值；不得按正向结果改选主配置。

## 5. Evidence 与兼容

- [x] 5.1 将设计中的六个新 CSV 接入现有 structured evidence 协议，补齐 summary/manifest 的输入、版本、单位、参数、质量状态和不可用原因。
- [x] 5.2 保留原文件/字段/旧 verdict 含义，对 legacy matched/CI 增加说明；测试新开关关闭时旧无 trace、单 trace 和普通调用兼容。
- [x] 5.3 测试共享 writer 的文件清单与序列化、发布失败传播，确保不新增 logger/writer/launcher 的 diagnosis 特例。

## 6. 聚焦验证

- [x] 6.1 运行新增分解、标准化、数据配对和 evidence 的内存 pytest，以及受影响的既有 diagnosis/artifact 测试。
- [x] 6.2 用 Hydra compose 验证开关独立/组合、输入缺失、local/W&B 引用与默认旧配置；不在单元测试运行真实 experiment 或 Trainer。
- [x] 6.3 对脚本执行 shell 语法及参数测试，覆盖两种 flag、空值、路径空格、错误输入与额外 override 优先级。
- [x] 6.4 运行 `openspec validate analyze-tail-search-ranking-loss --strict` 和 diff 检查，确认没有改变 decoder、checkpoint 或标准 bundle。

## 7. 四设置重分析与决策

- [x] 7.1 在执行前保存主配置和输入身份清单，复用下表的 evaluation、seed 42、K10/K50 Artifact；记录代码版本及前置未提交实现的可追溯信息。
- [x] 7.2 经根脚本和统一 `src.main` 执行四组新 diagnosis，显式开启两个分析模块，声明 calibration_statistics_ready=true；记录新 W&B run 与完整 evidence，不重跑 inference。
- [x] 7.3 核对六状态守恒、原始与新 Top10 计算一致、oracle 边界、来源身份和有效支持；补充 K50 截断 Top10 的正式 evidence。
- [x] 7.4 执行预声明分箱敏感性并呈现全部设置，记录支持不足与方向变化，不能隐藏失败配置或覆盖旧 run。
- [x] 7.5 形成当前提案目录中的结果报告和跨设置 probe_candidate/stop/inconclusive 审阅记录，分别讨论候选空间、排名与风险比较可信度；不自动创建或实现方法提案。
- [x] 7.6 在结果报告中附研究仓库状态交接清单：保留 H2 pilot No-Go、记录新分析、暂停未经支持的 H3、暂缓全矩阵；不跨项目改写历史材料。

## 8. 已知输入清单

以下是已有 Artifact 的运行 ID，不是新实验结果。统一使用 `wandb://<run-id>`，recommendation 与 prefix trace 由 role 分别解析；dataset 使用用户传入的数据根目录。

| 数据集 / SID | fixed recommendation + trace | widened recommendation + trace | SID | embedding | checkpoint identity |
|---|---|---|---|---|---|
| Beauty RKMeans | m4h0geda | 56rrarps | 4vyi4o6w | 3jtt9mpa | 26qh50do |
| Beauty RVQ | r94ut5sv | k7mr3xlk | d2hhqdic | 3jtt9mpa | ye9u9yj7 |
| Sports RKMeans | 5024wy48 | 4x3steyq | 3narllqy | psec3u5i | 129w8p0r |
| Sports RVQ | lzttakbs | n6svfy9g | ykntf4ve | psec3u5i | 49ote174 |

运行前先完成 1–6。新配置/flag 尚未实现，本文件不提供可被误执行的现成启动命令；实现阶段必须在真实参数 compose 和脚本验证后补充四条完整命令。
