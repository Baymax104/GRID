# CoPMRec v0 / v1 / v1.1 / v2 / v3 实现与运行

2026-10-04用户随后授权将首层Max、后续Mass的候选改动实现为从真实v0派生的v3。v3已新增独立组件与训练/推理入口，训练混合NLL和候选搜索共用该聚合，直接继承v0全部训练配置、dense验证选点及content终排；从零双卡命令见 [v3说明](copmrec-v3.md)。全部推荐参数可训练，无新增head或ranking loss。完整训练仍由用户手动开始，效果尚未知。

此前方向2 [真实v0 content bad case分析](copmrec-v3-content-bad-cases.md) 保留为历史证据。本次用户决定将当前重点转为候选搜索；下方“没有v3组件”的表述是当时阶段快照，已由本条实施决定取代。既有v0/v1/v2入口及关闭预算保留。

2026-10-04用户结束v2迭代，开始v3。v3与v1/v2平级、独立基于真实v0，当前仅完成 [收益转化卡点分析](copmrec-v3-conversion-bottlenecks.md)，没有v3运行组件或启动命令。v2负向结果、入口和产物保留，下方v2.x说明为此前版本规划，已由结束决定取代。

2026-10-04按用户澄清完成 [v0真实配置恢复与v2对齐](copmrec-v0-config-unification.md)：BMX-116原方法就是v0，当前v0入口恢复全evaluation dense验证、testing hybrid推理和500验证间隔；v2统一evaluation/testing，保留融合hybrid选点。v1/v1.1的selection/audit与2500间隔显式隔离，resolved行为不变。

2026-10-04新增 [独立基于v0的v2](copmrec-v2.md)，与v1平级；v1迭代结束于v1.1，后续新路线为v2.x。v2采用d-only head、自然候选尺度约束、固定lambda0.01/beta0.5。v1/v1.1推荐组件与已结束版本行为保留，推荐收益未知。

2026-10-04按用户要求删除v1.2并退回v1.1，保留四路head与校准排序权重。当前有效入口与命令见 [v1.1说明](copmrec-v1-1.md)，已撤回实现仅留 [历史记录](copmrec-v1-2.md)。

日期：2026-10-03。v1 已实现并完成聚焦验证，完整训练与收益评价尚未执行。基础 CoPMRec（包括 BMX-116 的既有方法结果）记为 v0，历史 W&B 结果不重写。

后续状态：v1已由用户启动正式训练（run `uzmkrfoa`），本说明下方的实现阶段结果保留为历史记录。性能排查与零更新优化验证见 [v1性能诊断](copmrec-v1-performance-diagnosis.md) 和 [v1.1实现与完整训练命令](copmrec-v1-1.md)。v1.1将训练/推理候选评分跨用户批量化，默认全局chunk256，全部参数继续可训练；新增独立入口，保留v1。用户已启动v1.1正式训练 `f91njtjx`，其联合优化诊断见 [设计诊断](copmrec-f91njtjx-design-diagnosis.md)；2026-10-04新增排序权重配置及 [有界验证](copmrec-loss-weight-verification.md)，完整推荐收益尚未确认。

## 版本边界

| 项目 | v0：基础版本 | v1：联合相关性版本 |
|---|---|---|
| 内容前缀概率 | 合法子树 mass 与生成概率逐层混合 | 保留 |
| 推荐参数 | 共享 encoder、内容投影、SID decoder、可学习融合参数 | 上述参数加候选相关性 head，全部可训练 |
| 训练目标 | SID CE + content CE + mixed NLL | 基础三项权重1；候选列表 CE 权重可配，默认1 |
| 最终 hybrid 排序 | content 分数 | content + 当前候选条件相关性残差 |
| 训练排序样本 | 无新增排序目标 | 每个 rank 每微批均匀抽 4 例，覆盖完整 training 人群 |
| 训练候选 | 原基础监督 | 当前 beam20 + 全部 cold + content Top20 + training 正例，去重 |
| 推理候选 | beam20 + 全部 cold | 保持，不注入真实目标 |

v1 相关性 head 为 `Linear(4d,128) -> GELU -> Linear(128,1)`。输入是用户、内容、逐元素交互和当前 decoder 读完完整候选 SID 后的末位状态。候选对按 64 分片；同一步带梯度的 encoder/内容投影复用，没有冻结推荐特征缓存。输出层零初始化，初始最终分数等于 content 分数；首步残差上游梯度为零，基础目标仍训练主干。

v0 实现与旧 `liger-joint-mixture-v1` checkpoint 保持兼容，该旧协议字符串与此次 CoPMRec 方法版本号是不同字段。新增 checkpoint 记录 `copmrec_version`；v1 还保存相关性结构、训练契约、可选预训练来源指纹和固定评价历史契约，拒绝直接将 v0 当作 v1 恢复。

## 入口与评价协议

版本化入口是 `copmrec_v0_train.sh` / `copmrec_v0_inference.sh`、v1/v1.1和v2同名脚本与Hydra experiment，统一通过 `src.main`。原 `liger_joint_train.sh` / `liger_joint_inference.sh` 继续保留，与当前v0的数据/评分/训练协议一致，版本化入口仅改变运行名称与产物标识等元信息。

v0按原全evaluation dense Top10的`val/ndcg@10`选best，在testing上做hybrid预测。v2使用相同evaluation/testing及500验证间隔，以自身最终融合hybrid评分选best。v1/v1.1继续从evaluation生成原哈希selection/audit划分，validation只用selection，inference默认只用audit，以各自最终hybrid分数选best；不将这组已结束版本的曲线当成统一数据后的匹配对照。

各版默认训练50k optimizer更新，AdamW lr3e-4、wd0.035，warmup2500，单optimizer连续训练；每卡batch128、累积1，双卡有效batch256。v0/v2每500微批验证，v1/v1.1仍每2500微批验证。v1/v1.1/v2每rank每微批抽4个排序样本，v0没有新增排序目标；GPU数、累积和batch必须记录，验证间隔按微批计，不按卡数除。

v1/v1.1的evaluation按rank分片且不补重复用户，v2使用与v0相同的FileDataModule按文件分配；v1/v1.1/v2关闭DDP buffer广播，Recall/NDCG在compute时同步总和与用户数，v0保留原配置。输出使用共享writer的keyed model output bundle；candidate trace的hybrid排名是最终分数，dense排名仍是content诊断。加入相关性的版本不适用旧“同分数子集排名不能差于全目录”的上界。

## 手动训练与推理

以下从 node1 仓库根目录执行。命令是手动入口，本轮未执行完整训练。GPU2、3是既有用户选择；开始时由用户按实际占用安排。

```bash
cd /data3/weizhenyu/projects/GRID
export PATH="$HOME/.local/bin:$PATH"
CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 bash copmrec_v1_train.sh \
  --dataset beauty --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --devices '[0,1]' --seed 42 \
  --notes 'CoPMRec v1：全参数候选相关性联合训练，固定 selection/audit'
```

匹配 v0 开发对照使用相同数据、上游、GPU、batch、seed与更新预算，改脚本为 `copmrec_v0_train.sh` 并记录对应 notes。两者的部署评分不同，best按各自最终部署链路选择，不依据audit选择checkpoint。

脚本支持显式 `--dry-run`、两种 notes 写法、额外 Hydra override 和 `NPROC_PER_NODE`，默认 `UV_NO_SYNC=1`。完整训练命令不默认dry-run；源快照、W&B配置与上游lineage继续由统一launcher记录。

训练完成后，从该运行的 best Artifact（`selection=best`、`monitor=val/ndcg@10`、`mode=max`）取得实际checkpoint引用，再手动执行 `copmrec_v1_inference.sh`，传入同样的数据/SID/embedding和 `--checkpoint "$COPMREC_V1_BEST_CKPT"`。当前没有v1 best，不能用旧v0或`last.ckpt`代替。

正式testing需要明确改变输入目录和范围：末尾追加 `evaluation_data_dir=data/beauty/testing evaluation_partition=all`。这不是默认audit，也不自动授予正式评价运行预算。

## 可选 v0 预训练初始化

默认随机初始化推荐模型。需要预训练时，训练命令追加 `pretrained_checkpoint_path="<真实v0 checkpoint引用>"`；可以是本地路径或带role/file的W&B引用。组件加载基础权重并校验目录和参数结构，初始化head输出为零；载入后仍全参数共同训练。checkpoint记录来源SHA256和v0 global_step，不复制teacher、不加载v0 optimizer。

恢复v1训练使用 `ckpt_path="<真实v1 checkpoint引用>"`，由Lightning恢复model/optimizer/scheduler；初始化与恢复是不同操作。比较预算必须计入预训练，不能把预训练v0加50k继续训练当作从零50k的匹配条件。

v1/v1.1支持 `model.root.ranking_loss_weight=<有限正数>`。v1组件及Python API默认1；2026-10-04用户采纳校准方案后，v1.1组件默认值改为0.05726763550972437。恢复时默认严格核验该权重；若有意继续训练并改变权重，须显式追加 `model.root.allow_ranking_loss_reweighting=true`，checkpoint记录原权重、新权重和来源步数，其他契约仍严格核验。推理应传入与训练一致的ranking权重以通过恢复检查；该权重只缩放训练loss，不直接缩放推理head分数。旧等权v1.1 checkpoint须显式传权重1。

## 验证与限制

本地120项聚焦测试通过，2项Linux Gloo检查因Windows传输环境跳过；覆盖零初始化与v0基础损失等价、完整SID末位、当前主干排序梯度、全部参数optimizer覆盖、标签独立、候选去重/cold、分片等价、checkpoint/optimizer恢复、初始化来源、数据划分、Hydra、脚本语法/quoting/空值/错误/override。node1验证13项通过，包含两进程连续2次更新的参数一致性，以及2人/1人非等长分片的全局Recall/NDCG分母。

Ruff与 `openspec validate add-copmrec-v1-relevance --strict` 通过。Mutagen flush成功，三个会话connected/Watching、无conflict；[20个运行文件指纹](evidence/copmrec-v1-20261003/runtime-files.json)与node1逐文件相同，旧退役入口仍未恢复。原始探针标量见 [生产探针记录](evidence/copmrec-v1-20261003/production-probe.json)。

生产模型零更新探针使用真实Beauty training的128条记录，上游SID dq77e3wo、内容3jtt9mpa，GPU1，12101商品、33cold、4排序样本；157组推荐参数全部可训练且梯度有限，2人Top10合法且无重复。v1一次前向+反向峰值allocated约3.103GiB，耗时约0.587秒；基础目标路径约2.372GiB/0.827秒。两次顺序单次测量包含不同warmup状态和dropout，**不能据此宣称v1更快**；不包含optimizer状态、完整数据装配、验证全体用户、DDP通信或训练轨迹。探针optimizer更新0、W&B run0，没有生成正式初始化checkpoint。

本轮是实现与契约验证，不是收益证据。最终仍需相对匹配v0及LIGER dense评价Recall/NDCG、新增/损失命中与位置变化。旧已关闭阶段不重置、旧scratch预算不转移，完整实验由用户手动开始。设计和固定判断协议见 [研究设计](../../research/docs/2026-10-03-copmrec-conversion-redesign.md)。

## 启动换行修复

2026-10-03 用户报告第2行 `set pipefail` 的 invalid option name。实际检查本地和node1字节确认四个版本化脚本均为CRLF，Linux Bash将回车当作选项名的一部分。此前Git Bash参数测试未发现此问题，补充了独立的原始字节回归检查。

四个脚本已转换为LF，新增 `.gitattributes` 的 `*.sh text eol=lf` 规则。32项配置/脚本测试和Ruff通过；Mutagen flush后三个会话Watching、无conflict。node1 Linux Bash分别验证四个脚本的语法及双进程参数转发（使用uv替身，不执行训练），20运行文件指纹一致，见 [修复后核验记录](evidence/copmrec-v1-20261003/runtime-files-lf-fix.json)。前一份runtime-files.json保留为修复前快照。

原完整训练命令可以重新执行，无须修改训练参数。本修复未启动训练或创建W&B run。
