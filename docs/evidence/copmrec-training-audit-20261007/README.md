# BMX-116 训练有效性与 Testing 命令审计

已登记的6个seed42/200训练通过固定预算与best产物核对；seed2026三个单元尚未启动。本次没有启动训练、Validation或Testing。

| Issue | Dataset | Seed | Train run | Best step | Validation NDCG@10 |
| -- | -- | -- | -- | -- | -- |
| BMX-120 | beauty | 42 | gshpyn49 | 47500 | 0.05268507 |
| BMX-119 | sports | 42 | jk4zk19n | 45000 | 0.03070792 |
| BMX-17 | toys | 42 | hs72xkan | 48500 | 0.05557501 |
| BMX-121 | beauty | 200 | lx1j76gv | 45500 | 0.05201440 |
| BMX-15 | sports | 200 | 8csakcss | 48500 | 0.03091753 |
| BMX-18 | toys | 200 | usnr8ary | 43000 | 0.05665846 |

核对范围：6个run finished、退出码0、50k正常停止；DDP2/global256/FP32、固定学习率与日程、从头初始化、v5.3四项loss和learned alpha一致。每run 100个非空Validation点恰为500..50000；选取最高NDCG的首次出现，与best Artifact metadata及checkpoint global_step一致。表中为Validation指标，不能当Testing结果。

源码tar逐文件哈希、归档SHA256、source aggregate及local-workspace origin一致，六run源码相同。SID/content Artifact version/digest已记录，同数据集两seed一致。best文件MD5与Artifact manifest一致，SHA256已固定；真实正式模型在CPU上通过on_load_checkpoint和strict state_dict，检查正式版本/seed/目录身份、完整优化器与scheduler状态、cold residual及有限权重。

last.ckpt并非50k终态，实际保存step见verification.json；完成50k使用完整validation曲线、trainer/global_step=49999零起始日志及max_steps停止信息证明，不用last.ckpt替代选点。没有基于该审计宣称训练已经收敛。原始数据文件未在训练时留独立逐文件manifest，此次目录身份检查覆盖SID/content及catalog契约，不能反向补造训练时raw-data字节快照。

每个BMX-*-testing.sh为独立单卡Testing命令，含不可变Artifact v0 URI、精确文件名、SHA256、seed和独立输出目录。通过Bash launcher参数stub及Hydra compose，未调用真实推理。执行后仍需输出合法性、split/user keys、独立指标复算核验，issue保持In Progress。

证据：audit.json为实时W&B配置/summary/完整validation history/Artifact元数据及checkpoint内容；verification.json为加载与哈希回执；prepared-commands.json为最终命令和配置验证。
