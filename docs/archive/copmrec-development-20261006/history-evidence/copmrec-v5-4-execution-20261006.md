# v5.4 固定验证运行记录

2026-10-06 用户明确批准“批准运行 v5.4”。本次仅新增1次从头连续50000更新训练＋1次单卡完整Validation，累计上限5train／250k／5Val；新Testing、seed43和扫描均为0。原四次实验的账本和封口证据不变。

v5.4 从 v5.2 派生，只取消 history 侧显式 residual，保留 catalog residual、原三 loss、learned alpha 和原50k训练配方。本次完整 Validation 明确负向，停止这个固定结构，保留 v5.2 主方法。它仍为共享编码器联合训练模型，不能称为恢复native query；单次结构对照不证明所有 history residual 都有收益。

## 启动与已通过检查

- 本地64项核心、80项配置与脚本检查，以及114项编排／审计／登记检查通过，独立审阅通过。
- 官方Mutagen同步通过，三个session均Watching且无conflict；实际运行源414文件，SHA `ae36e28d86d5a4c47c63d4f57b7a7753e2fe483168e9e1be62c414b47c6687c6`，旧408文件逐项SHA不变。
- 生产CPU预检确认11,031,809个可训练参数、165个state tensors、两个optimizer参数组、初始optimizer state entries=0；真实21字段契约与配置／checkpoint writer一致，没有checkpoint初始化。
- 物理GPU5、6→local[0,1]的一步smoke正常exit0，实际两个rank、1次更新、0个W&B run。smoke checkpoint不用于正式训练。
- 唯一正式job在 `2026-10-06T11:29:01Z` 启动，PID `87017`，目录 `logs/autonomous/copmrec_unified_catalog_only_residual50k_train_candidate42_20261006T112901281573Z`，物理GPU5、6→local[0,1]。正式W&B run为[cv5wwhck](https://wandb.ai/baymaxam/GRID/runs/cv5wwhck)，初始实际配置和source snapshot核验通过。

训练 `cv5wwhck` 已正常 finished／exit0，完整历史覆盖100个raw Validation点（500–50000），实际完成50000更新；累计五次训练共250000更新已核验。首次最大raw `val/ndcg@10` 对应46500步。own-best与saved-last均保存到46500步，完整参数、154份optimizer moments与scheduler只核到这个保存步骤；没有50000步终态full-state checkpoint。50k实际预算由完整历史与max_steps终止日志另行证明，不能把saved-last重标为50000步。

唯一完整 Validation `y9earebw` 使用物理GPU **5 → local[0]**、一个进程，正常 finished／exit0。主审计对175个原始文件／22363个用户、合法且唯一Top10、输入／标签／有效历史资格、实际消费的own-best checkpoint与414文件source逐项核验，并从原始预测重算下列指标。Lightning日志字段名为`test/*`，实际数据目录和产物split均为 **Validation**；本轮没有新增Testing。

## 实际训练配置与复现材料

以下来自正式run `cv5wwhck` 的实际resolved config，与生产CPU契约一致；不是用fixture生成的训练配置。

| 项目 | 实际配置 |
| --- | --- |
| 初始化 | seed42，推荐模型随机初始化，gate及residual从零开始；`ckpt_path=null` |
| 训练预算 | 连续50000 optimizer updates，`max_epochs=-1`，无early stopping |
| 并行与batch | DDP两进程，物理GPU5、6→local[0,1]；每卡128，accumulate=1，global256 |
| 数值与梯度 | FP32（`32-true`），gradient clip=1 |
| optimizer | AdamW，base LR=0.0003，residual LR=0.002，weight decay=0.035 |
| scheduler | warmup2500，cosine总步数50000，min ratio=0 |
| loss | SID CE、full-catalog dense CE、legal-prefix-mass mixture NLL，权重均1 |
| alpha与residual | learned alpha；scale=1；history residual=false，catalog residual=true，cold residual按seen mask为0 |
| 训练内Validation | 每500更新一次raw full-catalog dense Validation；选首个最大`val/ndcg@10` |
| 数据 | Beauty，同冻结SID `dq77e3wo` 和content embedding `3jtt9mpa`；history最多20 items |
| 后续部署 | own-best checkpoint，history eligibility下的full-catalog dense Top10，单GPU |

实际完整命令保存在[train-candidate42.sh](evidence/copmrec-unified-catalog-only-residual-50k-20261006/train-candidate42.sh)，从仓库根目录调用根训练脚本，最终入口为`uv run torchrun --nproc_per_node=2 -m src.main`。完整Hydra overrides含实际模型生成的21字段契约；[preparation](evidence/copmrec-unified-catalog-only-residual-50k-20261006/preparation-candidate42.json)同时保存argv、resolved config及414文件SHA映射。

实际推理使用的own-best引用为：

```text
wandb://baymaxam/GRID/cv5wwhck?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=046500.ckpt
```

文件SHA256为`cbb6972822c60341b8ea3963b86a72f7dbb9ea8444c2fd51c0f34c6037f2fdf6`。完整单卡命令保存在[val-candidate42.sh](evidence/copmrec-unified-catalog-only-residual-50k-20261006/val-candidate42.sh)，与[实际推理job](evidence/copmrec-unified-catalog-only-residual-50k-20261006/job-val-candidate42.json)的argv一致。训练内raw指标、后续history eligibility指标和checkpoint保存步数分别记录。

## 完整 Validation 结果与 bad case

| 固定方法 | Validation run | Recall@10 | NDCG@10 | R10相对native | N10相对native |
| --- | --- | ---: | ---: | ---: | ---: |
| LIGER dense | wdms8w77 | 0.0969458481 | 0.0540488986 | — | — |
| 保留的v5.2 | d7lcftto | 0.1014622367 | 0.0610582814 | +4.66% | +12.97% |
| v5.4 | y9earebw | 0.0903724903 | 0.0503776379 | −6.78% | −6.79% |

v5.4相对native的配对绝对差95%CI：R10 `[-0.00974936, -0.00339847]`，N10 `[-0.00550020, -0.00185040]`。相对v5.2，R10 **−10.93%**、N10 **−17.49%**，对应CI为`[-0.01426463, -0.00800429]`和`[-0.01274063, -0.00874462]`，同样均负。CI来自固定预测的22363个配对用户、PCG64 seed42／2000次bootstrap，是未校正的逐指标区间，不能解释为多次方案搜索后的独立确认。

- 相对native新增581／丢失728个Top10命中，净−147；共同命中位置的NDCG贡献为+0.000146，不足抵消命中损失。
- 与v5.2各自对native的统计相比，丢失命中数确实从824降到728（少96），但新增命中数从925降到581（少344），合计净命中少248。固定结构没有维持原新增收益，不能仅凭丢失数减少认定改善；这些是计数差，不表示96个或344个具体用户集合相互包含。
- 相对v5.2新增558／丢失806，净−248；共同命中1463个，其中381个上升、634个下降，位置变化贡献NDCG **−0.004044**。总NDCG损失中，新增与丢失贡献合计−0.006637，另有共同命中排序损失。
- 相对v5.2，Top5命中净−298，6–10名命中净+50；退步包含原前部命中的丢失与后移，不能只归为候选覆盖不足。
- 51个cold-target用户中命中4个，但错误cold Top10占位为784次（v5.2为44次）。这些是有限支持集的描述，不能证明cold占位导致其余命中丢失，也不据此推算Top11恢复。

以上分解只使用已有预测统计，不生成新候选或分数。两两汇总无法还原三方法逐用户四象限，本轮未拼造该统计。取消history显式residual会改变联合训练的梯度与学得表示；本次负结果支持在当前配方保留history+catalog共享residual，但具体表示、优化或校准原因仍未知。

## 五次固定实验的取舍与预算

同一50k问题累计结果如下，均为seed42、各自按raw Val N10选own-best后的完整Validation；不是等FLOPs、等墙钟时间或等搜索预算的证明，也不补齐历史native训练source缺口。

| 版本 | 已验证更新 | R10相对native | N10相对native | 取舍 |
| --- | ---: | ---: | ---: | --- |
| v5 | 50000 | +3.60% | +11.29% | NDCG部分收益；R10 CI跨0 |
| v5.1 | 50000 | −1.61% | +3.32% | 停止固定0.5／0.5双目录；两项CI跨0 |
| v5.2 | 50000 | +4.66% | +12.97% | 保留主方法；两项CI正，R10未达8% |
| v5.3 | 50000 | +7.06% | +13.88% | 保留部分证据；相对v5.2增量CI跨0，停止固定辅助CE |
| v5.4 | 50000 | −6.78% | −6.79% | 两项CI负，停止固定catalog-only结构 |

核心可反驳预测是：只取消history显式residual能减少seen命中损失，并保留catalog residual收益。本次观察反驳了这个固定结构在当前配方下的改善预测。既有v5.2部分收益仍有效，不能把本次负结果泛化为全部residual路线失败。实际CP／source／输入／输出审计未发现阻断这次取舍的实现问题；精确退步原因未知，不需要追加穷尽性排查才作停止决定。

新增1train／50000更新／1完整Val已用完；累计 **5train／250000更新／5完整Val**，Validation启动attempt共6次，其中旧阶段1次在产生预测前失败，独立保留。新Testing／seed43／参数、checkpoint或权重扫描均0，剩余授权训练与完整Val均0。本次不追加新实验或新预算请求。

native Recall@10和NDCG@10均至少+8%，且两项paired绝对95%CI下界为正的门禁保持，v5.2仅为描述参考；本次未通过。整体可复现目标仍未完成，成对seed43复现和冻结后Testing仍未运行；仅闭合v5.4固定验证阶段，不将整体goal标为完成。

## 可复核证据

- [实际授权与登记](evidence/copmrec-unified-catalog-only-residual-50k-20261006/stage-registration.json)
- [生产CPU预检](evidence/copmrec-unified-catalog-only-residual-50k-20261006/preflight-candidate42.json)
- [双卡smoke验证](evidence/copmrec-unified-catalog-only-residual-50k-20261006/smoke-verification.json)
- [实现与实际前置验证](evidence/copmrec-unified-catalog-only-residual-50k-20261006/implementation-verification.json)
- [唯一训练句柄](evidence/copmrec-unified-catalog-only-residual-50k-20261006/job-train-candidate42.json)
- [累计预算](evidence/copmrec-unified-catalog-only-residual-50k-20261006/cumulative-budget-latest.json)
- [实际启动回执](evidence/copmrec-unified-catalog-only-residual-50k-20261006/progress-training-start-receipt.json)
- [训练终态主审计](evidence/copmrec-unified-catalog-only-residual-50k-20261006/training-candidate42.json)
- [训练附加独立复核](evidence/copmrec-unified-catalog-only-residual-50k-20261006/training-independent-review.json)
- [完整Validation主审计](evidence/copmrec-unified-catalog-only-residual-50k-20261006/inference-val-candidate42.json)
- [Validation附加独立复核](evidence/copmrec-unified-catalog-only-residual-50k-20261006/inference-independent-review.json)
- [已有输出的bad-case分解](evidence/copmrec-unified-catalog-only-residual-50k-20261006/bad-case-analysis.json)
- [固定结构停止决定](evidence/copmrec-unified-catalog-only-residual-50k-20261006/stage-decision.json)
- [实际成本与阶段闭合回执](evidence/copmrec-unified-catalog-only-residual-50k-20261006/progress-validation-complete-receipt.json)

训练及Validation的附加独立复核均通过。本地复核重新核验绑定JSON、预算／身份／契约、指标与配对算术及414个runtime文件；远端checkpoint、预测PT、175个raw文件、source归档与新bootstrap的字节／指标结论复用SHA绑定的主审计，没有冒称二次读取远端或重新运行bootstrap。
