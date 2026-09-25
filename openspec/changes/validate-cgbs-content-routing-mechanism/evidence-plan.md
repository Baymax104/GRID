# CGBS 内容路由机制：有界证据判定

> 最新signal结果：exact相对trained为−0.332%，相对shuffled_exact为+5.648%（预定配对区间高于零）。优先保留原型与辅助监督，下一候选收敛为轻量状态门控；尚未实施或验证。见 [c-signal-outcome.md](c-signal-outcome.md)。

> 2026-09-16 更新：C/trained 与 C/off 已完成并核验，NDCG 点估计 +2.668%，预定聚类区间跨零，仍未超过 A；见 [c-screen-outcome.md](c-screen-outcome.md)。D 完全辅助梯度隔离退化，下一步仅交付原定 C signal 的两次冻结参考，尚未启动。

> 最新状态：B/C 已于2026-09-16完成，C 尚未超过 A。结果与下一步命令见 [training-outcome.md](training-outcome.md)。下文保留实验启动前的假设与预定判定规则，不能将其中“暂无 B/C”读作当前状态。

> 后续研究范围：用户要求继续固定CGBS及核心假设，检查复杂分支的调参、修正与必要模块。见 [refinement-review.md](refinement-review.md)。当前版本未通过晋级门槛的事实保持不变，但不再据此推断整个方法家族停止；下一优先候选是辅助梯度隔离。

## 当前结论与可复用证据

2026-09-16：**机制尚未确认；不是已证伪，也不是已支持。** 现有数据支持内容初始化值得保留、早期条件评分是重要失败位置，但没有同初始化的辅助监督控制 B 与完整分支 C。旧 Full 不是 A 加内容分支。

只读 W&B 查询确认目前不存在 `content_init_aux` / `content_init_full` run。原始核对记录见 `tmp/cgbs_mechanism_baseline.json`，查询代码为 `tmp/check_cgbs_mechanism_baseline.py`。

| 复用项 | 已确认身份或数值 |
|---|---|
| A 训练 | `baymaxam/GRID/5g3wpbg7`，finished，token_content_init |
| A best checkpoint | `tiger_catalog_grounded_token_content_init_beauty_train-checkpoint:v0` |
| checkpoint 文件 | `checkpoint_epoch=000_step=019000.ckpt` |
| A best val NDCG@10 | 0.0425697267；独立推理复算约 0.042576，不能混用两种精度口径 |
| SID | 上游 run `4vyi4o6w`；`rkmeans_inference-semantic-id:v2` |
| 内容向量 | 上游 run `3jtt9mpa`；`sem_embeds_inference-semantic-embedding:v5` |
| A 总体独立结果 | inference `cgbs-eval-5g3wpbg7-1789305926`；diagnosis `n5rv1dgm`，来自已有 outcome 清单 |

在线逐字段对比确认 A 与当前 `token_content_init` compose 的 model.root、train/val dataloader、seed、ckpt_path、beam、sequence length、层数、codebook、上游引用、max_steps、验证间隔、precision、strategy、checkpoint monitor 全部相等。复用仍要求 node1 数据未被替换、B/C 的实际 resolved config 与 Artifact lineage 一致；仅路径一致不证明文件内容一致。

已有现象：Beauty A NDCG@10=0.042576，旧 Full=0.041344；旧 Full 对 Mask CE 的 Tail 第2层 teacher-forcing rank 17.32→15.71，但 survival 6.967%→6.805%、命中40→39。Sports A 的4660个 Tail 目标中4467个在前两层丢失。这些都不能证明额外内容评分可以修复 A 的剩余错误。

## 唯一待验证假设

内容初始化之后，用户历史与分支内物品的内容匹配仍存在未被生成评分充分利用的互补信息。把该信息用于剪枝前的分支评分，能够在控制辅助监督后增加真实目标路径存活，并提高最终推荐净收益。

不是静态 SID 表达能力不可能、语义不可逆丢失、多原型必需或长尾提升的既定结论。

## 第一阶段：只补两个 Beauty 训练

| 条件 | 初始化 | 辅助 item CE | 在线混合 |
|---|---|---|---|
| A token_content_init | 内容 | 无 | 无 |
| B content_init_aux | 同 A | 0.1 | 无 |
| C content_init_full | 同 A | 0.1 | 同旧 Full |

共同协议：seed42、20k optimizer steps、每设备batch128、双卡有效batch256、Adam lr0.0005、验证每500步、best val NDCG@10、beam10、PCA128、四原型、temperature0.1、alpha初值0.1上界0.5。B/C 从头训练，不恢复 A optimizer 或 checkpoint。B 的额外模块不参与 beam；C 对 A/B 唯一额外主干干预为内容混合及其训练梯度。

在 node1 仓库根目录手动执行：

```bash
bash ./tiger_catalog_grounded_mechanism_train.sh \
  --data-dir data/beauty \
  --notes "CGBS mechanism qualification; same content init; isolate auxiliary supervision and pre-pruning content scoring"
```

脚本只按顺序运行 B、C，两次均固定物理GPU0/1。若仅一个条件失败，使用 `--condition b` 或 `--condition c`，不要重跑已完成条件。支持 `--dry-run`，默认完整训练；额外 Hydra override 置后，但改变预算、数据或初始化后不得作为本协议的正式对比。

## 第二阶段：先筛选，再按需要展开干预

训练完成先核验 W&B 实际配置、上游 Artifact、best checkpoint、完整步数。历史评估集已用于选型，本轮仅开发证据。不要将 summary 最后一次 val 当作 best，不把 batch/step 或采样差异混入方法收益。

推理使用明确训练 run 的 checkpoint 输出，示例中的 RUN_ID 必须替换为完成后核验的真实 ID：

```bash
bash ./tiger_catalog_grounded_mechanism_inference.sh \
  --data-dir data/beauty --condition b --stage screen \
  --checkpoint-path 'wandb://baymaxam/GRID/B_RUN_ID?role=checkpoint&file=checkpoint_*.ckpt'

bash ./tiger_catalog_grounded_mechanism_inference.sh \
  --data-dir data/beauty --condition c --stage screen \
  --checkpoint-path 'wandb://baymaxam/GRID/C_RUN_ID?role=checkpoint&file=checkpoint_*.ckpt'
```

screen 只输出 B trained、C trained/off，共3次单卡推理；不会自行展开 signal 或训练矩阵。预测与 prefix trace 按 user_id 对齐，复用既有 diagnosis 入口计算同口径推荐结果。C/off 是共适应模型的推理消融，必须与 B/C 训练对照共同解释。

只有需要区分“无内容互补信号”和“原型近似损失”时，才手动执行对应 checkpoint 的 `--stage signal`：exact 与 shuffled_exact 两个固定干预。对 B 优先检查信号存在性，对 C 检查压缩误差；无需默认把两者全跑。

- exact：逐物品 logsumexp，C 使用 checkpoint 的每层 alpha；B 固定0.1。
- shuffled_exact：同样的精确聚合、同alpha、固定seed42置乱，分支大小不变。
- rerank：同 token-only beam 的候选后重排，C 使用各层alpha均值，B固定0.1；没有候选并集。与 off 比較 Hit@10 必须相同，可比较 NDCG；prefix trace 禁用。它与逐层混合的函数形式不同，只是实用替代方案对照。
- 所有干预使用已有统一 inference 入口及共享 writer，训练 run 不变；prefix trace metadata 记录 mode/seed/alpha来源。resolved config 与 run notes 也记录干预。

精确聚合不是标签 oracle，也不是效用上界；任何模式不得按标签决定是否开启分支。精确模式比原型更慢的运行时间不能作为 CGBS 线上效率结果。

## 证据链及固定判定规则

1. **完整性先决条件**：相同evaluation用户、目标、SID/content身份、训练协议和完整目录；任何不一致先排查，不进入收益比较。未更改数据文件的前提需明确。
2. **辅助监督控制**：A→B 与 B→C 分开报告。主要指标 NDCG@10，辅助 Hit@10、逐用户新增/丢失命中；报告 A/B/C 绝对值与配对差值，不能仅比较旧 Full。
3. **作用位置**：同正确父前缀的目标概率、合法rank，与真实beam第1/2层存活分别统计；配对报告 rescue 与 damage，连接最终命中。主结果统计所有用户，按训练频次 Head/Mid/Tail 分组，不能只选择方法救回的错误样本。
4. **不确定性**：统一按前两层SID prefix 聚类的配对bootstrap，固定seed42、2000次，报告95%区间；组内过少item时报告绝对数。它估计用户/前缀抽样不确定性，不包含训练种子不确定性。
5. **进入确认阶段**：C 的 NDCG 同时超过 A/B，C−B 的配对区间支持正增益，Hit不下降，且对应干预的早期存活净改善传递到最终结果。还需记录额外训练/推理成本；通过才申请多seed/第二数据集/独立最终评估，而非当场宣称论文成立。
6. **辅助监督解释全部收益**：B有收益，C没有可靠额外收益，则不支持在线评分主机制；停止该分支的结构叠加。
7. **近似问题**：精确内容在固定协议下有净改善而原型没有，才允许针对近似误差提出一次明确修复；同时用精确置乱对照排除仅分支大小带来的效果。精确结果仍须观察误伤。
8. **局部指标不传递**：只改善局部rank，实际存活或最终结果未改善，机制链不成立。
9. **不确定或阴性**：区间跨零时标记未确定；不自动追加seed/sweep。固定查询/alpha下精确评分无互补收益时，结束当前实现的机制资格，不能外推为所有内容方法不可能。

总体收益若集中Head，只能讨论总体推荐质量；没有跨域Tail证据就不恢复长尾论文主张。显著性、效应大小、成本和相关工作差异需同时评估。

## 工程状态与科学状态

实现、轻量测试、代码同步的完成不代表机制被确认。完整 B/C 训练与冻结干预仍需用户手动执行，科学任务保留为未完成。

2026-09-16 工程验证完成：两个 CGBS 聚焦测试文件合计 **125 passed**；Ruff、脚本语法/参数验证、Hydra compose、OpenSpec strict、diff whitespace 检查通过。Mutagen flush 成功，四个 session 均 `Watching for changes`、无 conflict。没有自动启动 GPU 训练、推理或诊断，没有创建新 W&B run，没有提交 Git。
