# CoPMRec v5.4：仅目录使用 residual 的本地实现

## 当前状态

2026-10-06，本地可逆实现和聚焦检查已完成，独立静态审阅通过；推荐效果尚未验证。当前主方案仍为 v5.2，v5.3 的部分正向证据和停止固定辅助 CE 的决定保持。

已批准的累计额度为 4 次训练／200000 更新／4 次完整 Validation，已全部用完。v5.4 新增 1 次随机起点 50000 更新＋1 次单卡完整 Validation 的问题仍待用户答复，未分配、保留或启动额度；本次没有远端同步、生产 CPU probe、DDP smoke、正式训练或完整 Validation。新 Testing／seed43 pair／扫描均为 0。

原[固定 proposal](copmrec-v5-4-catalog-only-residual-proposal-20261006.md)保留形成时的快照，不回写为已实现状态；本文件记录其后的实际本地实现。

## 固定结构

v5.4 从 `UnifiedFullCatalogCECoPMRec`（v5.2）派生，只移除 history 中显式的 residual 加法。目录表示仍为投影后的内容加 seen residual；cold residual 仍为零。

- `item_content_residual` 返回 `None`，原 history 编码路径因此不再读取 residual。
- 新 `dense_logits` 保持原投影分块和调用顺序，显式调用 `CollaborativeResidualCoPMRec.item_content_residual`，保留目录 residual。
- 继承原 SID CE、full-catalog content CE、legal-prefix mixture NLL，三项权重各为 1，alpha 仍学习。
- 不新增参数、构造器参数、teacher、第二 query 或辅助 CE；不修改旧父模块。

query 仍通过 SID embedding、内容历史、共享 encoder 和联合目标学习交互信息。取消显式 history residual 不等于冻结 native query，也没有证据证明原 history residual 有害。

两处 sharing 均准确声明 `catalog_after_content_projection`。checkpoint 的 unified contract 增加以下 exact3 placement；恢复时严格核字段集合、类型和值，并继承版本和原始输入契约检查。v5.2／v5.3 checkpoint 即使 tensor keys 相同，也不能作为 v5.4 恢复。

```yaml
protocol: copmrec-catalog-only-residual-v1
history_residual: false
catalog_residual: true
```

## 可执行入口与训练配方

- 模型：[unified_catalog_only_residual.py](../src/recommendation/liger/unified_catalog_only_residual.py)。
- 配置：[model](../configs/model/copmrec_unified_catalog_only_residual_50k.yaml)、[训练](../configs/experiment/copmrec_unified_catalog_only_residual_50k_train.yaml)、[推理](../configs/experiment/copmrec_unified_catalog_only_residual_50k_inference.yaml)。
- 根脚本：[训练](../copmrec_unified_catalog_only_residual_50k_train.sh)、[推理](../copmrec_unified_catalog_only_residual_50k_inference.sh)，均走 `src.main` 与对应 Hydra experiment。

薄配置继承 v5.2：随机初始化、共同连续 50000 更新，DDP2／per-GPU128／global256／FP32，AdamW 基础 peak LR 0.0003、residual peak LR 0.002、weight decay 0.035、warmup2500／cosine50000／min ratio0、clip1。每500更新进行 raw Validation，按 own raw dense NDCG@10 选择 checkpoint；部署为单进程、单 GPU、history-eligible 完整目录 dense scoring。

训练和推理配置及 writer metadata 记录同一 placement 和 v5.4 身份。训练 writer 提供 `unified_scratch_contract` 的真实属性回填通道，当前默认 `null`；未来正式预检须从实际生产模型取得并核验，不能用手填契约替代。脚本支持 notes 两种形式、显式 dry-run、quoted 参数及后置 Hydra overrides；训练拒绝 warm checkpoint，推理要求显式 own checkpoint。

## 实际本地验证

两位实现 owner 分别执行并报告 exit0，第三位 reviewer 独立读最终源码／规格／配置和测试，逐项核冻结 SHA，不重复测试。

| 检查 | 实际结果 |
|---|---|
| 新核心31例＋v5.2核心33例 | 64 passed，4 warnings，5.27s |
| 新配置／脚本40例＋v5.2原40例 | 80 passed，4 warnings，22.42s |
| Ruff check／format check | 核心命令各 exit0；配置批次 exit0，各检查成功输出 |
| Hydra compose／完整 resolve、Bash `-n` 与真实 shell 参数捕获 | 通过；捕获使用 `uv` stub，未执行真实 `src.main` |
| 第三方静态审阅 | 15项通过，3个 Python 文件纯编译，未进行模型生产预检 |
| OpenSpec strict | 通过，正式阶段任务仍未完成 |

核心测试覆盖零 residual 时 state／RNG／query／logits／三项 loss 与 v5.2 一致；非零 residual 时 history 不读取该表而目录分数变化；SID-only 到 residual 的直接 history 梯度消失，CE／mixture 目录梯度保留且 cold 梯度为零；内存 checkpoint 正常恢复及非法 placement／旧版本拒绝。配置测试覆盖 writer 身份、notes、dry-run、空值／错误输入、quoting、后置 override 和 checkpoint URI／SHA／单进程约束。

这些是 CPU 内存 fixture 和轻量配置检查，不是生产数据预检、推荐效果或复现证据。

## 本地来源与证据

根 agent 用项目现有 source-manifest 函数核实际本地字节：旧 v5.3 **408／408 runtime 文件 SHA 不变**，只增加新模型＋model config＋两 experiment＋两根脚本，共 **414** 文件。

```text
local_source_sha256: ae36e28d86d5a4c47c63d4f57b7a7753e2fe483168e9e1be62c414b47c6687c6
old408_source_sha256: d6c5000509d968caabf5b7d46b5c5be3889c0c9b5e857d60fe5cb1d4cf3b22a0
```

这是本地 manifest 核对，尚无本版本运行端 source archive／publication。独立审查：[实际 review](evidence/copmrec-catalog-only-residual-proposal-20261006/implementation-independent-review.json)；本地准备与状态写入收据见 [local-implementation-preparation.json](evidence/copmrec-unified-catalog-only-residual-50k-20261006/local-implementation-preparation.json)。旧账本、注册、结题收据及闭合 state 尾部保持原字节。

## 后续判定与边界

仅在明确新增额度后登记固定 1train50k＋1Val，官方同步，再进行真实 CPU 空 optimizer／DDP2 一步 smoke／唯一正式训练及 own-best 单卡完整 Validation。单模型的 50k 更新与 global256 保持，不等同 FLOPs、完整研究搜索成本或收敛证明。

主门禁仍是同 seed native LIGER 的 R10、N10 各相对提升至少8%，且两 paired 绝对95% CI 下界为正。v5.2 增量仅作描述，未新增 parent 硬门槛；明确部分收益可保留，未获有效增量则停止此固定结构，不自动扩展扫描或下一臂。即使 seed42 通过，配对复现和独立 Testing 仍须真实证据，整体目标不能提前标为完成。
