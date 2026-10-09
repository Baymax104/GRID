# BMX-117 正式启动回执（2026-10-08）

用户明确授权开始 BMX-117：每个实验在 node1 使用独立 tmux；需要训练的 issue 启动并报告 run ID，不持续监控；无训练且输入已具备的实验执行至完成。范围保持 Beauty / training seed42，5 train / 250k updates / 0 额外独立 Validation / 5 Testing；不增加 seed、数据集或参数扫描。

## 五臂训练

| Issue | Variant | W&B run | 独立 tmux | 物理 GPU → CUDA local | 端口 | 启动观测 step |
| -- | -- | -- | -- | -- | -- | -- |
| BMX-122 | no_mixture | [m3frgiim](https://wandb.ai/baymaxam/GRID/runs/m3frgiim) | copmrec_m3_bmx122_m3frgiim | 5,6 → [0,1] | 29621 | 149 |
| BMX-123 | no_residual | [m364gahb](https://wandb.ai/baymaxam/GRID/runs/m364gahb) | copmrec_m3_bmx123_m364gahb | 5,6 → [0,1] | 29622 | 99 |
| BMX-129 | no_native | [m3td2jxc](https://wandb.ai/baymaxam/GRID/runs/m3td2jxc) | copmrec_m3_bmx129_m3td2jxc | 6,7 → [0,1] | 29623 | 149 |
| BMX-142 | legal_generation | [m3odfrrh](https://wandb.ai/baymaxam/GRID/runs/m3odfrrh) | copmrec_m3_bmx142_m3odfrrh | 6,7 → [0,1] | 29624 | 99 |
| BMX-143 | joint_ce_replace | [m3kq1tuc](https://wandb.ai/baymaxam/GRID/runs/m3kq1tuc) | copmrec_m3_bmx143_m3kq1tuc | 5,7 → [0,1] | 29625 | 199 |

真实命令、run ID、端口、tmux、Hydra 日志目录见 [training-launch-specs.json](training-launch-specs.json)，从当前 Linear issue 的训练命令生成，只调整资源并显式加入 run ID / 输出目录。每个 wrapper 已在 node1 通过 bash 语法检查。

五 run 均已在 W&B 出现，group=`paper_ablation_copmrec_beauty`，variant/Beauty/seed42/scratch/50k/FP32/val500 正确。NCCL 两个 rank 已注册，global_step>0 且首批 loss 有限。见 [training-startup-audit.json](training-startup-audit.json) 和 [training-startup-check.json](training-startup-check.json)。这些是启动工程核验，未作为论文效果证据。完成有界启动核验后停止持续训练监控，training_completed=0、Testing_started/completed=0；最终 own-best / SHA / Testing 结果待完整训练后审计。

## 输入与源码

[preflight-summary.md](preflight-summary.md)、[preflight-node1.json](preflight-node1.json)、[preflight-local-match.json](preflight-local-match.json) 登记真实数据与文件身份：304份运行源码与本地逐字匹配，source_origin=verified；实际训练 source_sha256=`42f0d7c7dce0a09961e0c8c23f9ba35982abfb796d13952c3f17eaeba17c9f2e`，各训练 run 已发布独立 `grid-source-<run_id>:v0`。

Beauty training/evaluation/testing 各175分片，共525份 SHA256；manifest SHA256=`46fbe99589b809ec8b7b9c21a815e4757c7a20d57664342ba8e525ecad8cb0c2`，Testing175分片与正式 Full 主结果匹配。SID / embedding W&B digest 分别为 `19ace08283163fbe687287fbacaa3842` / `ab56af975eac589c27eed6094482cbd7`，远端实际缓存 MD5 与 Artifact manifest 对应文件匹配；两者12101唯一 item keys，shape [12101,4] / [12101,1024]。

## 机制执行与依赖

M1/BMX-144 等待五个变体的 Testing 输出，当前尚未运行。M3/BMX-146 的 A1/A4 分支等待各自最终 own-best，不使用 Full 或临时 checkpoint 替代。

M2/BMX-145 初次 run `ls14h0dw` 全量测量完成后，在 summary JSON 写入时因嵌套 Hydra DictConfig 序列化失败，未产生完整 manifest，不能计完成。原 run / tmux / 日志保留，状态不能只以 W&B finished 或 tmux wrapper 的退出标记判定。已修复诊断 input_references 与共享 writer metadata 的 OmegaConf 普通容器转换；真实嵌套 DictConfig / ListConfig 与插值测试覆盖，19个聚焦测试、Ruff、OpenSpec strict 通过；Mutagen flush 后三个 session 均 Watching，无 conflict。该修复仅涉及 metadata 序列化，新诊断 runtime source 身份与已启动训练分别记录，不能回填到历史训练。

M2 使用新 run [717vgnkn](https://wandb.ai/baymaxam/GRID/runs/717vgnkn) / tmux `copmrec_m2_residual_bmx145_20261008_retry1` 完整重跑：699 batch，N=22363，输出四份 bundle、六份 CSV、summary 和完整 manifest。见 [m2-residual-independent-audit.json](m2-residual-independent-audit.json)：V11 按业务 keys 逐元素复现既有 Full，四视图均无非法/重复/历史重叠预测；指标、预定切片、配对 CI 和数值交互独立复核。M2 的最终 Artifact 12项 file:// reference 已逐项核对 node1 路径、size 和 base64 MD5。

Full prefix run [j2rworuj](https://wandb.ai/baymaxam/GRID/runs/j2rworuj) / tmux `copmrec_prefix_full_bmx146_20261008` 已完成单卡测量，699 batch、N=22363、四深度共89452条记录。见 [prefix-full-independent-audit.json](prefix-full-independent-audit.json)：概率/NLL/熵/JS/rank、全量与预定切片的均值及分位数复核通过；Full 全局 alpha 实测0.9857481122，概率 mixture 最大绝对数值复算误差1.5517e-7。最终 Artifact 8项 file reference 的路径/size/base64-MD5已全部通过；该分支不等于 BMX-146 整个 issue 完成，A1/A4 来源仍待产生。

两份诊断 runtime 源码 SHA256 均为 `5e95cf7e87b28b730976b5a6d279acdc7aca3a270a08dd5e4519ce1fc6624867`，origin=verified。 [serialization-fix-source-audit.json](serialization-fix-source-audit.json) 确认与训练原始源码相比，304份文件中只改变诊断引用和共享 writer 两处序列化文件；训练 source 不被回填为新版本。分析 manifest 的 `source_sha256` 字段指来源 checkpoint SHA，runtime 源码身份以 code Artifact / source snapshot / 独立审计中的字段为准。

M2 固定 checkpoint 2×2 实证原值如下，历史/目录分别表示该位置共享残差的开关。该测量与 A2 重训回答不同问题，不判别好坏。

| View | 历史残差 | 目录残差 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| -- | -- | -- | -- | -- | -- | -- |
| V11 | on | on | 0.05334705 | 0.03659647 | 0.07776238 | 0.04443310 |
| V10 | on | off | 0.04605822 | 0.03038388 | 0.06948978 | 0.03789313 |
| V01 | off | on | 0.03322452 | 0.02220864 | 0.05352591 | 0.02866918 |
| V00 | off | off | 0.03291151 | 0.02062044 | 0.05433081 | 0.02746638 |

失败重跑的工程开销单列，不增加训练、数据集或 seed 预算。观测只登记实证原值、带符号差值、区间、N 和解释范围，不判别好坏。

## 最终执行快照

父任务 BMX-117 / Milestone 3 保持 In Progress；5训练已通过有界启动核验，未等待完整训练，Testing 仍未启动。正式 diagnosis 任务启动2、交付2（计划4），工程 attempt3（含首次失败）。BMX-145 完整交付；BMX-146 的 Full 1/3 分支交付、其余两份 own-best 等待；BMX-144 等五份 Testing。Linear 父任务、统一协议、milestone、五训练与机制 issue，以及四份研究文档均回填真实 run / 来源 / 完成边界。

最终 Linear 回读见 [final-linear-readback.json](final-linear-readback.json) 和 [m2-linear-readback.json](m2-linear-readback.json)。八个实验 issue 均保留统一十章节，五训练 In Progress、BMX-145 Done、BMX-146 In Progress、BMX-144 Todo；协议与 milestone 的运行、来源和依赖一致。四份研究文档的 YAML/状态及26个本地引用核对通过。

观测图已生成 PDF/PNG，并视觉检查标题、图例、刻度及范围说明；配对差值图横轴刻度已修正为不重叠。绘图只读取已完成审计的原始观测，没有额外训练或 forward：

- [M2 四视图实证指标 PDF](m2-four-view-metrics.pdf)
- [M2 配对差值与 pointwise CI PDF](m2-paired-differences.pdf)
- [Full 条件化前缀观测 PDF](prefix-full-observations.pdf)
