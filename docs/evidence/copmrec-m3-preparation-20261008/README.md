# CoPMRec M3 入口与命令准备回执（2026-10-08）

协议：`copmrec-m3-v53-20261008-v1`。用户授权补齐必要入口并验证可执行命令；没有启动正式训练、Testing、诊断或真实数据 dry-run。没有创建 Git commit。

## 交付

- 五个独立 scratch 变体：no_mixture、no_residual、no_native、legal_generation、joint_ce_replace。分别实现冻结参数、精确损失组合、优化器组和 checkpoint 身份；跨臂及 Full 加载拒绝。
- 根目录训练/推理入口：`copmrec_ablation_train.sh`（双卡）和 `copmrec_ablation_inference.sh`（单卡）。
- `copmrec_diagnosis.sh`：统一 `src.main` / `Trainer.test`，单卡执行 hits、residual、prefix；不更新来源模型。
- epoch structured writer 一次写入完整 JSON/CSV、keyed bundle 与 manifest；最终证据 Artifact 发布，中间缓存保持本地。
- Linear 8 个 issue 的 15 个 Bash 命令块已写入并逐字回读，十章节模板、Todo、父任务及 milestone 归属已核对。命令保存在 commands.json 和各 issue 的 commands.md。
- 父任务 BMX-117、Milestone 3 说明及协议文档已更新为 implementation=locally_validated；运行结果仍为空。

## 固定预算与来源

只做 Beauty / training seed42。累计5 train / 250,000 updates / 5 Testing / 0额外独立Validation；另4个checkpoint诊断任务，M1无forward。M2的V11复现核对有额外评分开销，不能把纸面“3新增视图”理解为只有3次矩阵评分。没有增加 seed、数据集、训练变体或参数扫描。

Full checkpoint：
`wandb://baymaxam/GRID/gshpyn49?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=047500.ckpt`
SHA256：`a64ca93d49b5bb9ead7b4af803091346ca62af5299cd89266ac5a2acce3894c4`。
Full Testing output：
`wandb://baymaxam/GRID/vosmuihm?role=recommendation_output&alias=v0&file=merged_predictions_tensor.pt`。

新臂 own-best/SHA 和五份 Testing 输出尚未产生，命令明确标记待填；必须填写本臂不可变引用，不能借用 Full 或 last.ckpt。测试只为 argv/compose stub 使用内存 fixture 引用，不生产或伪造正式 checkpoint。

## 命令与资源

沿用已有 issue 的 export、bash launcher、分行参数、quoted wandb URI、双引号变量展开及 notes 写法。

- 训练示例：物理GPU0,1 → CUDA local [0,1]，NPROC=2。
- Testing/诊断示例：物理GPU0 → local [0]，NPROC=1。
- 五臂训练/Testing group：`paper_ablation_copmrec_beauty`。
- 全部机制分析 group：`paper_mechanism_copmrec_beauty`。
- variant/seed/stage/analysis 独立记录于 W&B config/tags/notes。GPU/端口示例不表示已预订资源。

## 验证

- 250项相关回归检查通过；末次诊断改动后15项CPU/lineage聚焦检查通过。
- 15个原样 issue 命令的 Bash语法、参数stub及Hydra compose通过；没有调用完整experiment或真实Trainer。
- 真实 Hydra 模型装配使用内存catalog/checkpoint验证 Full/A1/A4 来源恢复；公共初值/RNG、损失组合、全有效参数梯度、冻结参数退出optimizer和完整状态恢复通过。
- M1独立列表位置指标、keys、合法/唯一/历史资格、加法分解；M2重新encode/V11复现、权重保持、cold固定query logit不变；M3概率/NLL/合法log-rank（包含概率下溢边界）检查通过。
- 原子 epoch writer、bundle/manifest 和 checkpoint SHA 身份验证通过。
- Ruff和 `openspec validate add-copmrec-m3-experiment-entrypoints --strict` 通过。
- Mutagen三会话 flush成功，均 Watching for changes、无 conflict，代码交至node1；没有修改远端虚拟环境。

当前验证范围为CPU、配置、脚本及来源字段。真实Linux GPU/NCCL和完整数据运行尚未执行。新实验的observed/delta/CI、运行资源均为null；回执只表示入口准备，不是方法效果证据。
