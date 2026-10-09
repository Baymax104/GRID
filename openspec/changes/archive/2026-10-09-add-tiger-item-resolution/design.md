## Context

现有 TIGER 提供历史编码、T5 decoder、训练 metric 与 keyed 输出；CGBS 提供已验证的内容读取和对照依据。MIR 使用独立模块，直接调用 decoder state，确保当前前缀状态只看到 BOS 与已生成前缀。现有辅助 writer 绑定旧 prefix validator，需增加可选校验函数才能写新 trace。

## Goals / Non-Goals

**Goals:** 完成可启动的九条件、两数据集、三 seed 完整方法矩阵；同目录、同初始化约定、同 40k 更新预算；最终 item 概率与近似质量可审计。

**Non-Goals:** 不宣称方法有效，不自动启动 GPU，不修改 baseline，不把本地 COBRA/WIDE 适配称为论文官方复现，不重训量化器。

## Decisions

### 模型与目录

新增 `TigerItemResolution(Tiger)`，复用初始化和优化器接口，覆盖训练、评价、predict。九个 arm 为 `mask_ce / token_content_init / dense / hybrid / cobra / earliest / depth2 / depth_gate / mir`。固定目录按 item key 对齐，保存 SID、内容、CSR 后代关系、合法子节点、内容指纹。固定 PCA 压缩与已有对照一致；结构 arm 使用同一 token 内容初始化和可学习内容投影，mask CE 保持随机 token 初始化。MIR 不增加每 item 自由参数。

### 概率与训练

前缀状态 h_p 经 query head 与局部 item 内容相似度构成 R。合法下一位分布为 T。`F_p(i)=g_p R_p(i)+(1-g_p)T(c_i|p)F_pc(i)`；目标路径上 logsumexp 精确计算 item NLL。根强制路由，单 item 桶提前终止，末语义层强制解析。最大局部桶为 128，末层超过上限时初始化失败，避免无终点分布。

MIR 使用状态条件 gate；depth_gate 只看深度；earliest/depth2 使用固定出口。训练前 2000/40000 步预热，之后主 item NLL 加 0.1 路由 CE、0.1 可行出口 CE。固定出口使用相同辅助目标，保持直接对照。纯 dense 与 hybrid 单独记录自己的目标。DDP 使用 find_unused_parameters，以适应不同 batch 的可用出口及预热。

### 预算推理

MIR/固定出口采用前沿展开并按 item 累加贡献。尚未形成 K 个正质量 item 时优先完成较深前沿，之后按到达质量展开；这一共同调度避免平坦首层耗尽预算仍没有完整候选。每用户默认最多 64 个状态、4096 次 item 打分；完整保留未展开质量，输出下界分数和充分 Top-K 证书。预算不足以产生 K 个正质量唯一 item 时明确失败，不补任意物品或静默加预算。可更改推理预算，不改变训练概率契约。decoder 状态按批处理、深度分组，不逐 item 调用 Transformer。

### 最近邻适配

COBRA 适配显式保留 sparse→conditional dense 训练和 BeamFusion；在统一 TIGER 编码与固定内容输入下运行，说明与原文交替 sparse/dense 输入、文本 encoder 训练的差异。它是匹配信息控制，不代表官方复现。WIDE 风格在 hybrid checkpoint 上先用 training split 的有标签序列做熵标定，再按阈值 wildcard 扩张并混合打分；标定产物保存 checkpoint/catalog 指纹，evaluation/testing 不参与阈值拟合。该对照单独记录候选/状态成本。

### Trace 和实验入口

新 trace 名 `item_resolution_trace`，记录每用户前缀节点、gate、到达/解析质量、出口贡献、状态数、item 打分数、剩余质量、Top-K 证书；不混入旧 prefix survival。共享 AuxiliaryTensorWriter 注入新 validator，保持默认兼容。推荐仍输出完整四位合法 SID bundle。Outcome diagnosis 使用现有显式 split 的链路，不传旧 prefix trace。

实验通过根脚本及 `src.main` 启动。两组队列分别 GPU 0–1/2–3，平分 54 个从头训练；单卡 inference/标定；支持按 seed/arm 选择和 print-only 命令预览，dry-run 仍传统一入口。保存 best 与 last，W&B config 记录 arm、seed、协议、预算和来源。

## Risks / Trade-offs

- 固定深度或 dense 已足够强 → 完整保留强对照，失败时停止自适应主张。
- 概率质量优先并非最优 Top-K 搜索 → 报告剩余质量、证书率和实际单卡成本。
- 小目录无法证明大规模效率 → 以总体质量为主，不预设加速。
- 训练与推理的门控不一致 → 禁止硬阈值替代边缘概率，精确枚举小目录测试一致性。
- 基线适配不等于原论文复现 → 名称、配置与报告均标记 adapted；正式稿需要复现完整性说明。
- 旧 outcome 使用 evaluation 做 checkpoint 选择 → 本轮按 evaluation 选择，testing 在协议冻结后单独运行，不混称独立 test。

## Migration Plan

新增配置/模块即可选择新实验；原实验继续使用旧 target。只在新辅助 writer 配置注入 validator。通过聚焦 CPU 测试、Hydra compose、shell 参数回归及 OpenSpec strict 后交付人工命令。

## Open Questions

效果、gate 是否退化和实际延迟只能由正式实验回答，不阻塞实现。验收默认总体 NDCG 两数据集相对最强简单结构对照均至少 +2%，三 seed 报告，且实测延迟不超过该对照 1.5 倍；该门槛是研究取舍，不是显著性结论。
