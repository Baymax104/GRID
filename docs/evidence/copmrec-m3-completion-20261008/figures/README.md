# CoPMRec Milestone 3 论文表图

表图使用 Beauty、训练 seed=42、22,363 个固定匹配 Testing 用户的已审计结果。Full 复用正式主结果，各消融臂使用其验证集原始 NDCG@10 首次最大值选择的 own-best checkpoint。文件仅呈现实证原值、带符号差及分解；没有优劣标记、方向性结果分类或按结果选择样本。

| 文件 stem | 内容 | 数据行数 |
|---|---|---:|
| `paper-six-arm-metrics` | Full/A1–A5 的 Recall@5/10、NDCG@5/10 原值；另有 Markdown、LaTeX 表 | 6 |
| `paper-five-arm-minus-full-ci` | 5 臂相对 Full 的四指标带符号差与 95% pointwise paired bootstrap CI | 20 |
| `paper-specified-pair-ci` | A4−A1、A5−A3 的四指标带符号差与同协议 CI | 8 |
| `paper-m1-ndcg-contributions` | 5 臂在 K=5/10 的 variant-only、full-only、both-hit NDCG 贡献 | 10 |
| `training-raw-validation-ndcg10` | 5 臂各 100 次原始验证观察，区分选中点与 50k 预算终点 | 500 |
| `training-raw-training-loss` | 5 臂原始记录的训练 loss，不平滑；不同臂目标函数不同 | 5,000 |

各 stem 都提供 CSV、独立 PNG、独立 PDF。CSV 保留完整浮点原值，表中的 8 位小数和图中显示值仅用于排版。Arm 映射：A1=`no_mixture`，A2=`no_residual`，A3=`no_native`，A4=`legal_generation`，A5=`joint_ce_replace`。

CI 来自已通过独立核验的配对用户 bootstrap，NumPy PCG64、bootstrap seed=42、2,000 次重采样、95% pointwise 区间；其含义限定于这些固定 run 的用户重采样，不是训练 seed 变动的区间，也没有多重比较校正。本地绘图只读取审计标量，没有重新 bootstrap 或模型 forward。M1 原始 summary 标量和独立双精度均值的核对沿用终验绝对容差 1e−7，CSV 未改写 summary 原值。

NDCG 分解相对于 Full、以全体匹配用户为分母。仅变体命中项保留变体 NDCG，仅 Full 命中项已经带负号，两者均命中项记录排名折扣之差；不再次对 full-only 取负。彩色条按贡献类别固定着色，菱形显示原始总差，三项在核验容差内相加为总差。

训练曲线使用 W&B 历史原始点，不插值、不平滑；own-best 与固定 50k 更新预算终点分开标记。预算执行完成不等同于收敛声明。`../resource-observations.json` 保存 W&B 原始 run runtime 秒与双卡分配时长换算，active GPU 时长和峰值显存没有记录，保留 null。

输入证据、checkpoint/output URI 与 SHA、全部产物哈希见 `paper-ablation-evidence-manifest.json`。PNG 的标签、图例、区间、零参考线和贡献分解已逐张视觉检查；PDF 从同一 Matplotlib Figure 对象独立导出。
