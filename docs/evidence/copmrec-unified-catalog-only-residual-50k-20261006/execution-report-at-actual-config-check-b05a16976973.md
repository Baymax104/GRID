# v5.4 固定验证运行记录

2026-10-06 用户明确批准“批准运行 v5.4”。本次仅新增1次从头连续50000更新训练＋1次单卡完整Validation，累计上限5train／250k／5Val；新Testing、seed43和扫描均为0。原四次实验的账本和封口证据不变。

v5.4 从 v5.2 派生，只取消 history 侧显式 residual，保留 catalog residual、原三 loss、learned alpha 和原50k训练配方。它仍为共享编码器联合训练模型，不能称为恢复native query；作用与推荐收益等待真实验证。

## 启动与已通过检查

- 本地64项核心、80项配置与脚本检查，以及114项编排／审计／登记检查通过，独立审阅通过。
- 官方Mutagen同步通过，三个session均Watching且无conflict；实际运行源414文件，SHA `ae36e28d86d5a4c47c63d4f57b7a7753e2fe483168e9e1be62c414b47c6687c6`，旧408文件逐项SHA不变。
- 生产CPU预检确认11,031,809个可训练参数、165个state tensors、两个optimizer参数组、初始optimizer state entries=0；真实21字段契约与配置／checkpoint writer一致，没有checkpoint初始化。
- 物理GPU5、6→local[0,1]的一步smoke正常exit0，实际两个rank、1次更新、0个W&B run。smoke checkpoint不用于正式训练。
- 唯一正式job在 `2026-10-06T11:29:01Z` 启动，PID `87017`，目录 `logs/autonomous/copmrec_unified_catalog_only_residual50k_train_candidate42_20261006T112901281573Z`，物理GPU5、6→local[0,1]。正式W&B run为[cv5wwhck](https://wandb.ai/baymaxam/GRID/runs/cv5wwhck)，初始实际配置和source snapshot核验通过。

本记录当前只证明初始化、smoke和正式启动；尚未证明正式50k终态、own-best checkpoint或推荐收益。累计训练已启动5次，承诺250000更新，已独立核验完成仍为前四次200000更新／4次完整Validation。

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

后续推理命令中的checkpoint必须来自本次最终own-best审计，保留producer、Artifact版本、真实文件名和文件SHA；待训练终态后补入。训练内raw指标、后续history eligibility指标和checkpoint保存步数分别记录，实际更新预算最终由全历史与终止日志证明。

## 后续固定验证

等待同一job正常结束，审计100个raw Validation点、own-best选择、checkpoint完整状态边界、两份输入与实际runtime source归档。随后只运行一次已批准的单卡完整Validation，对175文件／22363用户独立核算原始Top10，比较冻结的native seed42与v5.2。

native Recall@10和NDCG@10均至少+8%，且两项paired绝对95%CI下界为正的门禁保持。v5.2增量用于描述机制取舍；单seed通过仍不能单独证明完整可复现目标。若无明确改善，停止此固定结构；不自动追加训练、调参或Testing。

## 可复核证据

- [实际授权与登记](evidence/copmrec-unified-catalog-only-residual-50k-20261006/stage-registration.json)
- [生产CPU预检](evidence/copmrec-unified-catalog-only-residual-50k-20261006/preflight-candidate42.json)
- [双卡smoke验证](evidence/copmrec-unified-catalog-only-residual-50k-20261006/smoke-verification.json)
- [实现与实际前置验证](evidence/copmrec-unified-catalog-only-residual-50k-20261006/implementation-verification.json)
- [唯一训练句柄](evidence/copmrec-unified-catalog-only-residual-50k-20261006/job-train-candidate42.json)
- [累计预算](evidence/copmrec-unified-catalog-only-residual-50k-20261006/cumulative-budget-latest.json)
- [实际启动回执](evidence/copmrec-unified-catalog-only-residual-50k-20261006/progress-training-start-receipt.json)
