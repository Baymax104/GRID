## Context

既有 TIGER 使用完整 SID（包含去重层），训练 CE 对层取平均；生成已经先屏蔽非法子节点再 softmax。训练仍在全局 `L*K` 词表监督。新模块必须保持原模型、数据展开和 artifact 协议不变。完整实验由用户启动。

## Goals / Non-Goals

**Goals:** 实现具有明确来源、可复现固定内容索引的 CGBS；提供八个实验条件、训练/推荐输出与 prefix trace、两组队列以及单元级验收。

**Non-Goals:** 本变更不证明有效性，不重训量化器，不使用测试交互构建内容索引，不宣称 Hybrid 等同于 LIGER，不自动运行完整实验或提交代码。

## Decisions

### 1. 模块及数据边界

新建 `tiger_catalog_grounded`，继承 TIGER 以复用 backbone、优化器和 batch 契约，覆盖新条件的训练、生成和 trace。`original` 保留原训练与生成。新 data helper 通过现有 `load_model_output` 对齐 SID 与 embedding 的 item key，拒绝重复、缺失、非有限内容、非法 SID 或非唯一完整 SID。所有额外参数放在独立 model component。

### 2. 固定内容与多原型目录索引

内容先做确定性 PCA（默认 128 维；维度不足时保留可用维数），再逐 item L2 归一化。PCA 只访问 catalog 内容，不访问验证/测试交互；因此属于固定目录的 transductive 内容设置。保存均值、投影、内容、key、SID 和 SHA-256 身份。随机对照仅用局部 generator 打乱内容与 SID 的对应关系，不能扰动 backbone 初始化。

每个非空 SID 前缀最多建立 M=4 个原型。确定性 farthest-first 初始化后做少量 Lloyd 迭代；保存真实 cluster 均值、item 数量和最大半径。均值不再归一化。单原型条件 M=1。原型按前缀稀疏存储，避免对所有 K^L 路径建表。

### 3. 评分、训练与生成

对 masked encoder pooling 进行共享投影并归一化得 q，温度 tau=0.1。分支分数为 `logsumexp_m(log n_m + q·mu_m/tau)`，在当前合法 children 内 softmax 得 Q。每层混合 `P=(1-alpha_l)P0+alpha_l Q`，alpha 上限 0.5、初值 0.1，以 sigmoid 约束。P0 为合法 children 上的 token softmax。

所有新条件生成 CE 按层取平均，与原 TIGER 尺度一致；有辅助损失的条件添加 `0.1 * CE(q X^T/tau, target_item)`。研究文档中按层求和的符号在实施说明中明确换算，不能混用 lambda。混合层、内容投影参数仅在对应条件创建，避免 DDP unused parameters。

生成使用同一 P 和 log-space beam 累积。只保留合法路径，最终推荐必须唯一且属于 catalog。小目录或窄树不能以非法 SID 填充。Trace 的 teacher 概率和 beam 分数分别沿用既有字段语义，明确记录 arm、catalog 身份和 checkpoint 来源。

### 4. 条件定义

| 条件 | 训练 | 生成 |
|---|---|---|
| original | 原 TIGER CE | 原 TIGER beam |
| mask_ce | 合法分支 CE | 合法 token beam |
| token_content_init | 内容均值初始化 SID 输入 embedding，合法 CE | 合法 token beam |
| single_prototype | CGBS M=1 + 辅助 CE | CGBS beam |
| full | CGBS M=4 + 辅助 CE | CGBS beam |
| no_aux | CGBS M=4，无辅助 CE | CGBS beam |
| shuffled | 内容对应关系固定打乱，其他同 full | CGBS beam |
| hybrid | 合法 CE + 相同内容辅助 CE | token beam 与全目录内容 Top-B 合并后，以 token 序列概率与全目录内容概率的 0.5 混合排序 |

`token_content_init` 是透明的内容初始化对照，仅初始化 SID 输入 embedding、保留每层原 embedding 标准差；去重层不做语义初始化。它不是特定论文的完整复现。Hybrid 合并后重新 teacher-force 计算每个候选的 token 序列概率，不能只按内容分数重排，否则必然退化为纯 dense Top-B。Hybrid 的最终推荐含稠密候选，不能将 token beam trace 冒充最终候选 trace；该条件只输出推荐 bundle，显式关闭 prefix trace。

### 5. checkpoint 与实验协议

固定 bank 为 persistent buffers；checkpoint metadata 包含算法版本、arm、bank 身份、温度、原型数和混合设置，恢复必须匹配当前构造来源。新 checkpoint 不允许无校验地作为原 TIGER checkpoint 加载。配置记录原始 checkpoint 引用，底层继续由统一 launcher 解析。

初始矩阵 Beauty 八条件、Sports 四条件（original/mask_ce/full/hybrid），seed=42，从头训练，最多 20,000 step，共享 batch、优化器、验证周期和按 val/ndcg@10 选优。第一轮只用于决定是否进入多 seed、多数据集主实验；不能视为已收敛或显著有效。禁止以多次测试集反馈选参数。

训练默认不自动测试；推理要求显式 data split。先用 evaluation 做方向选择，冻结后 testing 做正式报告。两组队列分配 GPU 0–1 和 2–3，各顺序启动六个训练；训练完成后用户提供 run IDs，按最佳 checkpoint 启动推理/diagnosis。相同 data/SID/embedding 引用由命令传入，data-dir 无默认值，seed 默认 42。额外 Hydra overrides 放最后。

原 `DiagnosisDataset` 在没有 trace 时默认 testing。为支持 Hybrid 的 evaluation 推荐，在 `src/data/catalog_diagnosis.py` 增加独立子类，要求显式 data_split，并验证其与可选 trace 一致；通过独立 experiment 和薄 wrapper 脚本复用既有 diagnosis。不得把不同 checkpoint 的 trace 当 fixed/widened，也不得开启只适用于旧配额干预的 candidate-allocation-probe。

### 6. 验收与停止规则

验收 key 对齐、精确单 item 分支质量、原型计数、梯度有限、所有可训练参数获梯度、合法唯一 beam、trace 语义、checkpoint mismatch、全部 arm 实例化、Hydra 组合和 shell quoting。保持 baseline 文件零修改。

实验判定联合看 Overall/Head/Mid/Tail NDCG 与 Recall、命中用户净变化、Tail 候选可达性及时间/显存。full 必须超过 mask_ce 和 no_aux/token_content_init 的简单替代解释，并与 hybrid 比较质量及在线成本；不预设成功，不以单个 Tail 命中波动定方向。此后仅对胜出方案进入 3 seed 扩展，不继续滚动增加小权重试验。

## Risks / Trade-offs

- [原型近似偏差] → 保留 cluster 半径与计数；单元验证 exact leaf，报告 M=1 消融。Jensen 下界只描述 log-mass 近似，不保证排序改善。
- [内容与行为不一致] → shuffled、no_aux、hybrid 对照分别检验对齐、辅助训练和检索路线。
- [目录规模与构建时间] → 稀疏原型、固定 CPU 建库和 checkpoint buffers；第一版每进程建库，可在实测确认瓶颈后独立优化缓存。
- [硬件下 DDP 与显存] → CPU 真实小模型 backward 验证；完整 GPU 性能仍须用户首轮运行验证。
- [公平性] → 相同层平均 CE 尺度、seed、数据展开、预算和选优规则；原 baseline 与新合法 CE 的差别由 mask_ce 单独控制。

## Migration Plan

先落规格，再新增 loader/index/module/config/scripts/tests，最后聚焦验证。回退只需选择原 `experiment=tiger_train`；原路径不受影响。本变更不归档，待实际实验后用户决定。

## Open Questions

无阻塞实施的问题。方法效果、最优训练时长和真实在线成本属于待实验回答的问题。
