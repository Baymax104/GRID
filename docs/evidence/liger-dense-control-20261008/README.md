# BMX-147：LIGER dense 单卡内部对照完成证据

用户授权仅执行Beauty/seed42一个Testing。协议 `copmrec-internal-liger-dense-beauty42-v1`；[Linear issue](https://linear.app/baymax104/issue/BMX-147)已Done/有效，[W&B run](https://wandb.ai/baymaxam/GRID/runs/ldi54f1o) finished、exit=0。完成按来源和核验完整性，不按指标方向。

## 实际执行

Training=0，独立Validation=0，完整Testing=1。复用35ig0tz6正式own-best，step45000，URI `wandb://baymaxam/GRID/35ig0tz6?role=checkpoint&alias=v1&file=checkpoint_epoch=000_step=045000.ckpt`，SHA `8508e08e2a2cc9ea5d2bbc902aa4e2b6415c8a45b9c8a0ddc879728a6b66b43b`。不重新选点，不初始化新训练。
物理GPU1→local[0]；tmux `liger_dense_bmx147_ldi54f1o`；group `paper_internal_liger_beauty`。
运行目录：node1:/data3/weizhenyu/projects/GRID/logs/liger_dense_control_20261008/BMX-147_beauty_seed42_ldi54f1o。

## 完整指标与预定配对

| 来源 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| -- | --: | --: | --: | --: |
| CoPMRec Full | 0.05334705 | 0.03659647 | 0.07776238 | 0.04443310 |
| LIGER dense（本次，共同历史资格） | 0.05155838 | 0.03450915 | 0.07722577 | 0.04281558 |
| 原LIGER hybrid（原资格协议参照） | 0.03774091 | 0.02399962 | 0.05594062 | 0.02986749 |

| Full−dense | 差值 | 95% pointwise CI |
| -- | --: | -- |
| recall@5 | 0.00178867 | [-0.00116263, 0.00482941] |
| ndcg@5 | 0.00208732 | [-0.00003958, 0.00410931] |
| recall@10 | 0.00053660 | [-0.00263829, 0.00380092] |
| ndcg@10 | 0.00161752 | [-0.00043920, 0.00359993] |

相同22363名用户，user/target SHA `55e2602110f989565aa6c5fc00bef99107a17d3d11222c06c46874b9df50e016`。PCG64 seed42、2000rep、95% pointwise；未校正多重比较，不表示跨训练seed稳定性或等价。原hybrid保持原历史资格政策，其差值不作为单一候选范围的因果效应。

cold目标N=138：本次dense Recall@10=6/138，Full=0/138。记录小样本原值，不外推cold总体收益。

## 来源与完整性

输出[N,10,4]合法、唯一，历史重叠0；4metrics独立NumPy复算与W&B最大误差小于1e-8。Checkpoint文件大小/MD5/SHA和Artifact身份passed，checkpoint的item keys/SID/content/seen buffers与当前输入逐值一致。304文件runtime source SHA `5e95cf7e87b28b730976b5a6d279acdc7aca3a270a08dd5e4519ce1fc6624867`，origin=verified，archive内容与W&B manifest通过。训练保持原始来源，不回填历史源。

最终输出Artifact `liger_beauty_seed42_dense_history_excluded_inference-recommendation-output:v0`，digest `66c70135fda1fbb7dd506dad94844465`；URI `wandb://baymaxam/GRID/ldi54f1o?role=recommendation_output&alias=v0&file=merged_predictions_tensor.pt`。W&B最终输出为node1 file://引用，path/size/base64-MD5/SHA已核验，最终文件继续保留。W&B tracked runtime=31秒；active GPU时长及峰值显存未记录=null。

## 可追溯文件

- [完整独立审计](audit.json)：全部来源、输入分片SHA manifest、4metrics、2组配对差值/CI、cold原值、资源。
- [启动前文件身份](preflight.json)、[100点own-best选点核验](own-best-history-verification.json)、[完整实际命令](testing-command.sh)、[启动规格](launch-spec.json)、[独立tmux启动回执](launch-receipt.json)。
- [完整resolved Hydra](hydra-compose.yaml)、[脚本/Hydra组合核验](hydra-compose-check.json)。
- [启动控制](prepare-launch.py)、[只读独立审计控制](run-audit.py)、[只读远端审计](audit-remote.py)：模型运行仍唯一进入根脚本和src.main；审计不执行模型forward。
- [研究判断与结论边界](../../../../research/docs/copmrec-liger-dense-control-20261008.md)。
- [Linear十章内容与Done/有效最终回读](linear-final-readback.json)。

本次仅完成1/9内部dense条件，其余8个未启动。BMX-117五臂/250k训练、M3八issue与正式主矩阵/原LIGER hybrid完成事实保持不变。
