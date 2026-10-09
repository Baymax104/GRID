# 首批三个 CoPMRec 双卡 tmux 训练启动回执

2026-10-07用户授权在node1启动三个实验，每个双卡且必须使用tmux；随后明确允许共享显存充足的GPU。按计划选择Beauty/Sports/Toys的seed42，对应BMX-120/119/17。

| 数据集 | tmux会话 | 物理GPU | 端口 | 新W&B训练run |
|---|---|---|---:|---|
| Beauty | copmrec_beauty_s42_20261007 | 1,5 | 29501 | gshpyn49 |
| Sports | copmrec_sports_s42_20261007 | 3,6 | 29503 | jk4zk19n |
| Toys | copmrec_toys_s42_20261007 | 0,7 | 29505 | hs72xkan |

各组映射为CUDA local[0,1]，NPROC_PER_NODE=2；正式Linear训练命令的模型、输入、seed、group、预算及notes保持，仅调整物理GPU、独立端口与Hydra日志目录。

运行根目录为node1:/data3/weizhenyu/projects/GRID；日志在logs/copmrec_formal_tmux_20261007/<dataset>_seed42/train.log。tmux断开SSH后继续运行，运行结束时记录exit_code并保留shell会话供检查。

已确认三个tmux会话、各2个src.main工作进程、NCCL注册、两rank CUDA映射、六张GPU计算活动、v5.3正式模型/50k/FP32配置、正确group、上游Artifact lineage以及三个runtime源码快照包/manifest/runtime metadata。记录为训练运行中，不是完成或效果证据；不启动Testing或其余seed。

启动前Mutagen flush/status三会话Watching无conflict。首次远程启动因Windows文本管道CRLF在执行第一行前失败，未产生实验；修正为二进制LF传输后启动成功，失败回执保留于launch-receipt-before-lf-fix.json，不隐藏attempt。
