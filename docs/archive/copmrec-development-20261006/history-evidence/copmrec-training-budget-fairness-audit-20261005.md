# CoPMRec 最新版与 LIGER dense：训练预算及续训公平性审计

## 当前结论

当前保留的 v4.4 control history pool 与现有 LIGER dense **没有相同训练预算**。LIGER 与 CoPMRec v0 的初始训练各完成 50,000 optimizer updates；最终 CoPMRec 两成员所需的共享生产链随后增加 6,000、6,000、2,000 updates，完整实际成本为 **64,000 updates，比 LIGER 多 28%**。

完整 Testing 已验证的 R@10 +10.770122%、NDCG@10 +10.776636% 是相对固定旧 LIGER checkpoint、相同 history 资格下的结果。它仍然成立，但目前不能表述为“相同训练预算下超过 LIGER 10%”，也不能据此排除训练更久、optimizer/LR 重启或 checkpoint pooling 的贡献。

本次仅实时读取 W&B、复算已记录数据并检查可执行配置和既有实际 checkpoint 审计，**新增正式训练及推理均为 0**。没有实施新模型、修改运行配置或新增实验授权。

## 实际训练与权重来源

| 阶段 | 实际 run | 完整运行 updates | 被使用的阶段 checkpoint | 初始化 |
| --- | --- | ---: | ---: | --- |
| LIGER dense | `35ig0tz6` | 50,000 | 45,000 | 从头训练 |
| CoPMRec v0 | `7y54j4m6` | 50,000 | 48,000 | 从头训练 |
| v4 residual | `nj9elah1` | 6,000 | 6,000 | v0 best48000，仅权重 |
| v4.1 control | `l3zyr91b` | 6,000 | 6,000 | nj9 best6000，仅权重 |
| v4.4 control | `hbgj80bh` | 2,000 | 2,000 | l3 best6000，仅权重 |

这五个实际配置的训练 preprocessing 逐值相同，均使用同 training 路径及 SID、最多 32 个 causal 展开样本、最多 20 商品的有效输入、sequence length 80。各阶段均为双卡、每卡 batch128、gradient accumulation1，因此 global batch256；seed42、FP32、gradient clip1。

- LIGER 完整生产链：50,000 updates，对应 **12,800,000 次展开后目标样本呈现**。
- CoPMRec 最终共享生产链：50,000 + 6,000 + 6,000 + 2,000 = 64,000 updates，对应 **16,384,000 次展开后目标样本呈现**。
- 已消费权重的祖先：LIGER 为 45,000；CoPMRec source 为 48,000 + 6,000 = 54,000，continued 为 48,000 + 6,000 + 6,000 + 2,000 = 62,000。checkpoint 自己的 local step2000 不能代替其完整训练祖先。
- source 是 continued 的祖先快照，二者共享前缀，因此生产实际成本不能把 54,000 与 62,000 相加为 116,000。
- 先前“7 次训练 / 34,000 步”仅统计自主阶段的新增搜索成本，未包含 v0 的 50,000 步。加上 v0，该部分已知搜索支出为 84,000 步；它也不等于最终部署的 64,000 步生产链，更不是全部历史搜索成本。

“样本呈现”允许重复训练样本，不是唯一样本数或 epoch 数。相同 updates/global batch 不等于相同 FLOPs、GPU hours 或超参数搜索预算：CoPMRec 有额外 mixture objective、residual 参数，最终还对两个成员分别编码和评分。历史训练 runtime source archive 缺口不能由新推理 source snapshot 回填；本次对初始配置一致性的结论限于实际记录。

## LIGER 已出现平台，但尚未证明重启后不能提高

实时完整有界 scan 返回 LIGER 原训练全部 100 次 Val NDCG@10，每 500 updates 一次；不是从 W&B summary 或采样曲线猜测。该序列与 trainer/global_step 的对齐读数及被选 checkpoint metadata 完全一致：

| 原训练位置 | 记录的 Val NDCG@10 | 含义 |
| --- | ---: | --- |
| best45000 | 0.04646100476384163 | 全部 100 次验证最大值 |
| terminal50000 | 0.04637987166643143 | 相对 best 约 -0.174626% |
| best 后十次验证 | 均未超过 best45000 | 原日程末 5,000 updates 无新高 |

初始 LIGER/v0 使用 AdamW LR0.0003、WD0.035、warmup2500、cosine horizon50000、min_ratio0。按记录配置及当前同名 scheduler 公式计算，45k 处 LR约8.13e-6，50k 处为0；这是**配置推算值，不是历史实测 LR 日志**。旧训练没有完整 runtime byte archive，因此不将当前公式追认为旧运行的源码证明。

CoPMRec 后续每段仅恢复权重，`ckpt_path=null`，重新创建 AdamW 和 scheduler；backbone LR0.0001、residual LR0.002、WD0.035、warmup300、cosine horizon6000。最后一段只运行2000但仍保留6000的 horizon，checkpoint 实测两组 LR约7.961e-5 / 0.001592。因此“LIGER 接近零 LR 时无新高”不能排除“给予同样重启及额外训练后它也会提高”。目前只支持**原日程末期进入平台**，不足以完成用户要求的第一条证明。

![原始同预算训练曲线](evidence/copmrec-training-budget-20261005/original-training-curves.png)

另一个边界是阶段验证资格：nj9/l3 使用原 dense 无 history 排除，hbg 使用 history-excluded dense。因此跨阶段直接比较它们的训练 Val 指标，不是纯续训净收益。已有旧/new pool 的同政策独立评测可以描述增量，但它尚未匹配 LIGER 的训练预算。

## 推荐最小证明：匹配生产更新预算与两成员推理

优先选择用户提出的第二条路线。固定当前 CoPMRec，不再优化其 checkpoint、pool 权重或训练方法；为 LIGER 增加以下一个连续生产 pipeline，明确新增成本为 **3 个阶段 / 14,000 optimizer updates**，不是重置旧阶段预算。

| LIGER 新阶段 | 实际运行上限 | 权重初始化 | optimizer/scheduler | 阶段 Val checkpoint 选择 |
| --- | ---: | --- | --- | --- |
| A | 6,000 | 自己原 best45000 | fresh AdamW、LR1e-4、WD.035、warm300、horizon6000 | 原 dense、无 history 排除，Val N10 best |
| B | 6,000 | A 自己的 Val best | 同 A，重新创建 | 同 A |
| C | 2,000 | B 自己的 Val best | 同 A，horizon仍为6000 | history-excluded dense、Val N10 best |

公共条件保持 global batch256、seed42、FP32、同 causal32/有效历史20/sequence80、clip1、每1000 updates 验证、原 SID CE 与 content CE。LIGER 没有 CoPMRec residual 参数组，不给它凭空增加该模块；其原参数组使用相同 backbone LR 日程。每段仅恢复权重、optimizer state和scheduler重新初始化，不能用普通 trainer checkpoint resume 代替。

这样双方**完整生产更新成本均为64,000，目标样本呈现均为16,384,000**。各自采用自身验证 best，所以 LIGER 被选权重祖先最多59k，CoPMRec continued为62k；不能为了凑权重祖先步数偷换 LIGER 的best为48k。相同可用且实际执行的训练预算与 selected checkpoint 祖先步数须分别披露。

最终验证两个 LIGER 方案：C 单成员；A best + C best 各独立编码后固定0.5/0.5 full-catalog dense logit pool。两者都使用当前双方一致的有效输入 history 排除及稳定 catalog row 排序，不扫描 pair、权重或额外 LR。pool 对照可以匹配当前 CoPMRec 两成员结构；single 对照说明 pooling 对 baseline 的贡献。

以新两方案及已存在的原 LIGER 同政策 Validation 为候选，预先按 Val N10最高、再R10、再简单方案选择唯一 baseline。只对该 baseline 做一次新完整单卡 Testing；若选择已有原 baseline，则复用既有同政策 Testing。当前 CoPMRec 的 `ws2fx4oi` 原始 output、keys、标签和输入指纹可直接复用，无须重新挑方法或用 Testing 选点。

结论规则：同预算 baseline 下 CoPMRec 若仍明显提高，则支持“收益不只是额外 optimizer updates/样本呈现和两成员 pooling”；若增益缩小，报告真实剩余增益并收缩 10% 公平预算主张；若差值 CI跨0，不把它写成等效或无提升的证明。这个对照不证明全局优化收敛、同 FLOPs 或同超参数搜索成本。更完整论文证据还需要既定多 seed/数据集协议，不能由单个 Beauty/seed42 对照代替。

当前 native LIGER 没有训练用 weights-only continuation 参数，`HistoryExcludedLiger` 明确禁止 fit；直接给旧 train 命令加 max_steps/ckpt_path 不会得到上述实验。实施需要薄 continuation 适配，复用 native hook、strict state、统一 Artifact loader及 history helper，并在运行前验证新 optimizer/global step0。**本审计仅给出具体可审阅的设计，尚未实施或启动这三个阶段；原7/34k及Testing3/3已封存，不自动追加。**

## 可追溯证据

- [本次结构化审计](evidence/copmrec-training-budget-20261005/audit-summary.json)：预算 ledger、全部轻量断言、完整历史检查、已有证据文件SHA256。
- [实时 W&B 读取快照](evidence/copmrec-training-budget-20261005/wandb-query-snapshot.json)与[完整 Val 历史及步数对齐](evidence/copmrec-training-budget-20261005/training-history.json)。
- [nj9实际训练审计](evidence/copmrec-v4-autonomous-20261005/training-residual-lr20.json)、[真实optimizer/scheduler补证](evidence/copmrec-v4-autonomous-20261005/training-residual-lr20-supplement.json)、[l3实际审计](evidence/copmrec-v4-autonomous-20261005/training-bias-control.json)、[hbg实际审计](evidence/copmrec-v4-autonomous-20261005/training-eligible-control.json)。
- [既有完整同history Testing](evidence/copmrec-v4-autonomous-20261005/eligible-winner-testing-audit.json)及[此前结题报告](copmrec-v4-4-eligible-content-continuation-research.md)：数值结果与本次新识别的预算边界分别保留。
