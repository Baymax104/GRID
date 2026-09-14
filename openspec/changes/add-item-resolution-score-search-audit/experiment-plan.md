# MIR 六条完成训练分析与下一步取证

日期：2026-09-15。当前状态：用户停止自动矩阵并删除不完整run；重新只读查询确认仅保留以下6条finished训练，无运行中MIR实验。原54条件计划中的其余48条件暂停，不作为必须补跑任务。

## 1. 已有证据与结论

| 数据集 | arm | run | 最佳验证NDCG@10 | 最佳步数 |
|---|---|---|---:|---:|
| Beauty | MIR | wgx7944n | 0.05037848 | 38000 |
| Beauty | earliest | 9qt9hww3 | 0.05007041 | 36000 |
| Beauty | depth2 | m47u9t4k | 0.05151321 | 36000 |
| Sports | MIR | y2ilymxo | 0.02385600 | 40000 |
| Sports | earliest | l81ioxiz | 0.02334445 | 40000 |
| Sports | depth2 | 6jt7s9ro | 0.02545237 | 40000 |

六条均为seed42、40k更新、80次evaluation验证，best/last Artifact元数据已在进展核查中校验；本轮再次确认这些run全部保留且finished。W&B日志global_step采用零基索引，表中按实际更新次数显示。

MIR相对depth2：Beauty −2.20%，Sports −6.27%。MIR虽略优于earliest，但没有建立查询相关深度的必要性。训练曲线持续改善，不能描述为训练完全失败；Sports峰值在预算末端，也不能断言完全收敛。当前数据支持暂停大矩阵，不支持声称MIR有效或最终统计否定。

不能从 `train/gate_mean` 约0.56–0.58推断门是否退化：该均值还混合强制终止的层/单item桶，不等于查询自适应门的分布。现有验证记录也没有全目录目标排名、item概率捕获与真实出口责任，因此尚不能确认问题来自训练评分还是搜索近似。

## 2. 唯一下一步：复用4个best checkpoint

- Beauty：MIR `wandb://wgx7944n`，depth2 `wandb://m47u9t4k`。
- Sports：MIR `wandb://y2ilymxo`，depth2 `wandb://6jt7s9ro`。
- SID/embedding沿用：Beauty `4vyi4o6w`/`3jtt9mpa`；Sports `3narllqy`/`psec3u5i`。
- 仅evaluation，每数据集按SHA256(20260915,user_key)选择最小128个用户，MIR/depth2完全配对。与标签、模型输出和文件遍历顺序无关。
- 每用户枚举全部目录item，分块128得到精确边缘概率、目标rank/logprob、目标出口责任和全局出口概率质量。
- 同时运行Q=64/128/256、S固定4096，记录搜索目标rank、Top-K、证书、剩余质量、对精确Top-K的重合和返回item的概率捕获比例。
- 四次推理串行使用单GPU，无训练、无验证循环，无额外seed。精确chunk控制显存；默认batch1/num_workers0用于稳定抽样。

完整分布需要遍历目录，仍有计算成本，但用户数固定128、checkpoint固定4个。没有真实GPU计时，因此不承诺具体分钟数。每模型遍历约128×目录item数的目标路径；Beauty约155万、Sports约235万条，按chunk128并行。相对对整套数据重复80次验证，范围显著缩小。

## 3. 运行命令

先将本次代码同步到服务器，在GRID仓库根目录运行以下**一个命令**，会依次产生4个取证run，始终只使用GPU0：

```bash
NPROC_PER_NODE=1 bash ./tiger_item_resolution_audit_suite.sh \
  --beauty-data-dir data/beauty \
  --sports-data-dir data/sports \
  --gpu 0 \
  --notes "MIR score versus search decision; frozen completed checkpoints; no new training"
```

若使用其他卡，只改 `--gpu`。末尾添加 `--print-only` 仅打印四条命令，不访问W&B、不执行模型。`--dry-run` 为统一入口小执行模式，不能计入正式证据。

显存不足时可以对**整个队列**追加 `audit_chunk_size=64`，只改变精确枚举分块；数学分布不变，数值误差有测试覆盖。如果必须缩小用户数，对整个队列统一追加 `audit_users=32` 并明确记录协议变化，不能对两模型使用不同样本后直接比较。

产物为每run的 `item_resolution_audit` Artifact，文件 `item_resolution_audit.pt`，本地路径位于Hydra输出的 `audit_evidence/`。这是取证bundle，不应作为旧prefix trace或全量recommendation结果传给Tail-SID diagnosis。前向编码、精确枚举、预算搜索计时分开记录。

## 4. 完成后如何作决定

先核对4个run、各128个用户、同数据集keys/labels/input_sha256/目录一致、checkpoint指纹不同且来源正确。所有全目录概率和出口质量应归一化；状态/评分次数不能超过所声明预算。

主分析使用配对目标rank（建议同时报告log-rank）、目标NLL、全目录分布以及搜索与精确排名差距。128用户的Hit/NDCG非常稀疏，不能用少数命中推导2%论文提升或统计显著性。精确rank优势与NLL优势需要分别报告，NLL更低不保证Top-K更好。

| 观察 | 资源与方法决策 |
|---|---|
| MIR精确目标rank在两数据集呈一致优势，Q64有明显漏失；增加Q能改善且成本可接受 | 保留评分结构，提出有针对性的搜索/预算匹配修正；仍不恢复54-run矩阵 |
| MIR精确排名与概率没有可信优势，或全局出口质量近乎固定深度且无收益 | 当前MIR主张停止投入，基于该失败机制重设计；不再追加seed或辅助loss扫参 |
| 优势仅在昂贵的精确枚举或更大成本下出现 | 当前质量—成本定位不通过；不能将精确枚举当可部署方法 |
| 样本结果混合、区间很宽 | 明确证据不足，维持大矩阵暂停；不自动开始更多小验证 |

Q变化时S保持4096。如果评分预算已饱和，Q增大无效不能证明不存在搜索误差；精确分布用于消除这一歧义。目标出口责任描述对真值的解释，全局出口质量描述模型输出分布，两者不能混称gate学到了某种机制。

## 5. 实施验收与边界

OpenSpec `add-item-resolution-score-search-audit` 已先提案再实现。新增独立模块及配置，未修改已有训练目标或数据流。

聚焦测试覆盖：精确分布与穷举搜索一致、分块一致、无标签抽样、键去重、改变真值不改变生成、共享writer往返、checkpoint加载、Hydra模型/数据装配、脚本语法/quoting与四run单卡预览。没有在本机启动完整GPU取证，待用户执行后再给出评分与搜索的最终归因。

本轮模型与配置/脚本共94项测试通过；补充证书字段后的4项取证回归再次通过（与94项重叠，不相加）。Ruff与OpenSpec strict通过。GPU时间/显存尚未验证，完整运行由用户启动。
