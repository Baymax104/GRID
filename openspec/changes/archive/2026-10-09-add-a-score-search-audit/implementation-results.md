# A 评分与搜索审计：实施及手动运行

## GPU 评分一致性修正

用户首次GPU运行在beam/teacher概率比较处报错。代码核查发现src.main全局启用`torch.set_float32_matmul_precision("medium")`，此前CPU测试默认highest；不同batch/序列长度的内部矩阵乘精度是优先排查因素，原错误没有误差数值，尚未在GPU复现确认根因。

审计现将encoder、目录teacher和原beam统一包在highest精度上下文，正常/异常退出均恢复原设置。保留atol=1e-4/rtol=0，追加最大误差、具体item的两条logp、用户key、chunk及设备诊断；证据必须注明highest。这是原搜索算法的FP32参考审计，不能声称与历史medium推理逐位一致；耗时也只代表该审计精度。真实GPU修复效果待用户重新运行。

## 当前状态

G0 已实现；真实 GPU 审计尚未运行。G1 完整 SID 排序训练尚未实施，须先根据 G0 决定是否推进。用户已采纳这个有条件的顺序。

- 模型：原 A，`token_content_init/full_content`，默认 Beauty seed42 最佳 checkpoint `5g3wpbg7`，19000 步。
- 上游：SID `4vyi4o6w`、语义向量 `3jtt9mpa`；使用显式文件引用，实际解析身份由既有 lineage callback 记录。
- 数据：evaluation 中按 `SHA256(20260918:user_key)` 选择固定 128 用户，与真值、遍历顺序无关；batch1，workers0，单进程。
- 评分：encoder 每用户一次，完整目录按 128 项分块，使用原 A 的合法条件概率累加；保留实际 beam10 结果，不改变搜索算法。
- 产物：仅审计 evidence，无新 checkpoint。保存每用户全目录 logp、beam keys/logp、目标排名及容差上下界、前缀生存、三段耗时、输入 hash、checkpoint 和目录指纹。

## 手动命令

远端运行前，在本地仓库根目录执行既有同步入口：

```powershell
./mutagen_sync.ps1 flush
./mutagen_sync.ps1 status
```

只有 flush 成功且三个 session 都是 `Watching for changes`、无 conflict，才在 node1 的 GRID 根目录执行下面一个命令。`data/beauty` 必须指向包含 `evaluation` 子目录的数据根；物理 GPU 按当前空闲情况选择。

```bash
bash ./tiger_a_score_search_audit.sh --data-dir data/beauty --gpu 0 --notes "G0: frozen A exact catalog versus beam10, fixed 128 evaluation users"
```

可先在同一个命令末尾加 `--dry-run` 检查真实链路；这是会加载模型和数据的 smoke run，不是打印命令，也不是完整 128 用户审计。实现验证没有运行它。额外 Hydra override 保留最后覆盖能力；例如显存不足可加 `audit_chunk_size=32`，不改变待评分目录。不要将减小 `audit_users` 的结果当作预定的 128 用户证据。

脚本 `--seed` 是本次执行随机种子；默认 checkpoint 始终是 A seed42，不会因该参数自动换成其他训练 seed。默认 notes 明示这一身份，改变输入引用时需同时检查 resolved config 和 lineage。

## 产物与判读

本地输出为 `${paths.output_dir}/audit_evidence/a_score_search_audit.pt`，由共享 `AuxiliaryTensorWriter` 发布到当前 run，Artifact type/role 均为 `a_score_search_audit`。无参数 checkpoint 输出；主要 tensor 体积约为 `128 × 目录项数 × 4 bytes`，另加目录身份等元数据。实际空间与耗时以运行结果为准。

完成后先核验：run finished、唯一用户数等于 `requested_users=128`、evaluation、checkpoint/目录身份与预期一致、validator 通过。validator 同时用于逐行和合并检查，允许小于 requested_users 的局部证据，因此仅有文件或校验通过不等于完整实验完成。

| 统计 | 定义与用途 |
|---|---|
| 确定搜索遗漏 | beam 未命中，且目标精确排名容差上界 ≤ 10；当前评分已能命中，但 beam 丢失 |
| 确定评分失败 | beam 未命中，且目标精确排名容差下界 > 10；仅精确搜索不能救回 |
| 边界不确定 | beam 未命中，且容差排名区间跨过 10；避免将同分或微小数值差归因于搜索 |
| beam 特有命中 | beam 命中，但稳定排序的精确 Top10 未命中；不能把精确 Top10 称作相关性上界 |
| Top10 重合率 | 同一用户实际 beam 与精确概率 Top10 的 item 集合交集 / 10 |
| 前缀首次失败 | 目标首次不在实际 beam 中的深度；用于定位，不单独证明评分或搜索因果 |

同时汇总 exact/beam Hit@10、NDCG@10、配对净增减，以及搜索遗漏/评分失败的绝对人数和分母。128 用户中的少量命中只能作为方向筛查；单纯“评分失败占多数”是常见现象，不能证明排序训练会有效。若精确评分明显救回目标，优先调查搜索；若恢复空间有限，再考虑 G1。命中太少或边界不清时明确记录不确定，不能据此宣布方法有效。

## 后续 G1 的边界

协议见 [design.md](design.md)：同一原 A 权重起点，CE 与 CE+采样完整 SID 排序两个分支，各 2000 更新、同 lr/优化器重置，每 500 步验证；必须同时对比 CE 续训和冻结 A，并记录训练成本。当前没有 G1 实现或启动脚本，避免先运行再选择判据。此排序项作为已知技术的工程探针，新颖性与效果均未确立。

## 验证

验证包括微型内存目录的概率归一化、CE 等价、分块一致、真实 beam 对齐、label 独立、原 checkpoint 严格加载与冻结保护、容差排名分类、损坏证据拒绝、writer 合并与重复 key、确定性抽样、Hydra 组件装配、Bash 语法和参数透传。未启动完整 experiment 或真实 Trainer 链路。

结果：新增两份测试文件共 32 项通过；原 A 与二层交互相关回归 106 项通过。Ruff check/format-check 和 `openspec validate add-a-score-search-audit --strict` 通过。checkpoint 兼容性采用原 A 类生成的内存 checkpoint 验证，真实远端 checkpoint 加载与 W&B 发布仍待手动 smoke/full run。
