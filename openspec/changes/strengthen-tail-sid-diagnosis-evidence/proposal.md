## Why

当前 Tail-SID diagnosis 已能证明 Beauty 上的 Tail item 存在 raw SID collision、深 prefix near-collision 与末层消歧负担，但现有 summary 缺少 Head/Mid/Tail/Tail-Cold 的同口径原始指标对照，且 `tail_damage` 混入预设 frequency gate，无法独立证明 tail-specific resolution damage。诊断也尚未把 item/prefix 风险与 TIGER 推荐错误关联或输出可供后续 repair/reranking 复用的结构化 Artifact，因此还不足以作为正式优化方法的证据基础。

## What Changes

- 将数据观测得到的 `raw_damage` 与人为设定的 frequency-aware `priority_score` 分离；研究假设检验只使用原始分量和 `raw_damage`，priority gate 仅用于后续处理排序。
- 为 Head、Mid、Tail、Tail-Cold 输出同口径的 collision、near-collision、prefix density、suffix burden、semantic mismatch、frequency-asymmetric overlap 与 damage 分布统计。
- 增加 Tail-Head、Tail-Mid、Tail-Tail 等 overlap 关系以及 head-dominated bucket、tail isolation deficit 等 tail-specific 非对称指标。
- 将 semantic compatibility 拆成全局随机 pair 阈值下的主 `semantic_mismatch` 与 bucket-relative outlier 敏感性口径，并记录确定性采样与阈值元数据。
- 新增可选 `recommendation_output_path`，从 keyed TIGER inference bundle 与 testing labels 计算 item-level hit/rank 结果，并分析 damage 与推荐错误在整体及 Tail 内的关系。
- 增加 frequency-bin matched analysis、Tail 比例与 damage/semantic 阈值敏感性，以及可复现的 bootstrap effect size/置信区间。
- 恢复结构化 diagnosis 输出，将 summary、group、item、prefix 和 sampled harmful-pair 结果写入本地分析目录并发布为 W&B diagnosis Artifact，供后续 SID repair 与生成重排消费。
- 为每次 diagnosis 输出明确的 evidence verdict，按预定义 go/no-go 标准区分 tail structural asymmetry、equal-risk tail vulnerability、generation-risk validity 与证据不足。
- 将现有模块职责与依赖方向作为硬约束：data 层只负责引用解析、keyed 输入与 label assembly；Tail-SID diagnosis 域只负责证据计算；common writer 只消费通用结构化 payload 并负责落盘/发布；lineage callback、metric callback、launcher 和 W&B logger 生命周期不得吸收 diagnosis 特例。
- 不修改 RKMeans、R-VQ、RQ-VAE 或 TIGER 训练逻辑，不在本 change 中实现 SID reassignment、量化 loss、beam reranking 或大规模多数据集训练。

## Capabilities

### New Capabilities

- `tail-sid-diagnosis-evidence-artifact`: 定义 diagnosis summary、group/item/prefix/pair 表的本地与 W&B Artifact 输出协议，以及供后续方法消费的稳定字段和 metadata。

### Modified Capabilities

- `tail-sid-resolution-diagnosis`: 将现有 Tail-only summary 扩展为无 gate 的分组证据、frequency-asymmetric overlap、双口径 semantic compatibility、可选推荐相关性、敏感性分析和 go/no-go verdict。

## Impact

- Affected code: `src/quantization/tail_sid_diagnosis/` 的 metric/context/result 计算，`src/data/` 中 diagnosis 输入与 testing-label 对齐，`src/common/writers/` 中可复用的分析 Artifact 输出装配。
- Affected config: `configs/experiment/tail_sid_diagnosis.yaml`、对应 data/model/callback/writer 配置与根目录 `tail_sid_diagnosis.sh` 参数透传。
- Affected tests: diagnosis 分组指标、semantic 阈值、推荐输出 key 对齐、bootstrap/sensitivity、writer/Artifact/lineage 与 Hydra compose 聚焦测试。
- External systems: W&B diagnosis run 将继续记录 summary，并新增一个包含结构化证据文件的 logged Artifact；可选 recommendation input 必须记录 upstream lineage。
- Dependencies: 优先使用现有 Python、PyTorch 与项目 I/O 能力实现，不为统计或表格输出引入 pandas、SciPy、PyArrow 或绘图库。
- Backward compatibility: 不提供 `recommendation_output_path` 时仍可运行纯 SID diagnosis；现有 `semantic_id_path`、可选 `embedding_path`、短/完整 W&B URI 与本地路径保持兼容。
- Architecture: 不允许 `src/common/` 或 `src/data/` 反向依赖 `src.quantization.tail_sid_diagnosis`；不允许 launcher/metric callback 增加 diagnosis 条件分支；不允许 writer 解析 W&B 输入、计算领域指标或管理 run 生命周期。
