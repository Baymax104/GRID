# seed 2026 训练审计与 Testing 命令

三个训练finished、退出码0；通过从头初始化、正式v5.3、DDP2/global256/FP32、50k固定预算核对。100次非空validation覆盖500..50000，首次最高raw NDCG@10与best checkpoint Artifact metadata一致。

| Issue | Dataset | Seed | Training run | Best step | Validation NDCG@10 |
| -- | -- | -- | -- | -- | -- |
| BMX-118 | beauty | 2026 | 8e9ji2s9 | 48000 | 0.05237612 |
| BMX-16 | sports | 2026 | 32ykmgdi | 47500 | 0.03089908 |
| BMX-19 | toys | 2026 | qpaw09y8 | 42000 | 0.05599444 |

已验证best真实文件与Artifact manifest MD5，固定SHA256；真实CoPMRec CPU on_load_checkpoint及strict state_dict通过，目录身份、有限参数、cold residual、正式seed/版本与优化器/scheduler契约一致。源码tar逐文件SHA256、aggregate hash、归档hash和local origin一致；SID/content输入Artifact lineage已记录。last.ckpt保存较早状态，不用于Testing；完成预算使用完整曲线和正常停止日志，未宣称收敛。训练时未保留raw-data逐文件快照，不反向补造。

三个BMX-*-testing.sh为可独立复制的单卡Testing命令：精确best v0 URI、文件名、SHA256、seed2026、testing split、统一group和独立输出目录。Bash launcher参数stub及Hydra compose通过，未调用真实推理、未创建新W&B run。

证据：audit.json、verification.json、prepared-commands.json、linear-readback.json。Testing和独立结果核验待完成，issue保持In Progress。
