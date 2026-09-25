## Context

用户已采纳 `E:/projects/research/ideas/2026-09-15-main-method-brir.md`。当前 MIR dense 以 decoder BOS 查询和 PCA 目录特征评分，CGBS 仍训练生成损失；二者均不满足 BRIR masked-mean/raw-content projection 契约，因此不自动迁移已有 checkpoint。

## Goals / Non-Goals

实现四种训练 arm、训练 split 标定、动态/固定/reference 检索、证据与手动入口。复用 TigerEncoder、现有序列预处理、MetricEngine、AuxiliaryTensorWriter。完整训练、论文效果和新颖性确认不属于实现验收。

## Decisions

- 独立 LightningModule；catalog 使用原始 keyed embedding，经可学习线性映射至128维，不执行 MIR PCA。历史仍使用固定 SID 输入及 TigerEncoder separator。
- `base` 训练 full-softmax；`dense`、`prefix_free`、`brir` 从同一明确 global_step 的 BRIR base checkpoint 初始化，重置优化器。所有分支持有冻结且 eval 状态的候选来源；dense 自身更新，残差组只更新两层 decoder/独立 token embedding/MLP。随机负例按输入历史 hash 和固定 seed 产生，与 dropout、batch 顺序及模型更新无关。
- 独立 `calibration` predict 任务：training 每用户最后一个已观测 next-item 历史，按 user key hash 取1024行；不含 evaluation/test，不读取标签选择用户或计算 gap。使用10/128分数间距的 midpoint median/2，绑定 base 权重与目录 fingerprint。全量/分支模型通过统一 data helper 加载。
- 用户指定训练使用0、1卡：训练支持单卡或双卡，双卡采用DDP并启用unused参数检测；标定、推理和审计仍限定单进程单卡。四组采用相同拓扑，完整实验用户启动。
- 动态检索按基础分数下降顺序分块，按前缀缓存decoder状态，分数上界加明确数值余量。预算未完成返回不足证据标记；稳定 item key 破同分。reference、fixed64/128/256、dynamic 在 audit 中独立运行并记录集合/顺序一致、item/prefix计数和同步耗时。只依据bound不能输出已验证证书。
- checkpoint 保存完整评分/训练契约与来源 fingerprint，拒绝旧 MIR、错目录、错步数、错标定。分支 checkpoint 恢复仍使用 ckpt_path；base_checkpoint_path 专用于初始化与候选来源，二者不可混同。
- 四组统一每卡microbatch8；单卡累积16次，双卡累积8次，有效batch均为128。分别每16000/8000个microbatch即1000优化步验证并保留checkpoint（包括1k/3k/5k）。使用一个save_top_k=-1的ModelCheckpoint兼容现有writer；best用于参考，三分支基础来源必须用20k last。训练脚本不串联自动矩阵，逐阶段给出命令。

## Risks / Trade-offs

- 界可能覆盖全目录 → 保留完整reference及实测成本，不声称次线性复杂度。
- 浮点 batch 形状影响边界 → 原始bound、exact集合、exact顺序分别记录，不声称经验余量为形式证明。
- 标定训练来源每用户一条，区别于训练增强32前缀 → 明确记录 sampling rule，不伪称整个增强训练分布的均匀样本。
- frozen teacher 增加 dense continuation 成本 → 单独记录，不声称同step同算力。
- 开发验证本身可能耗时 → 预先固定协议，不能静默缩小用户样本或预算。

## Migration Plan

独立组件上线，无旧checkpoint迁移。轻量测试和 strict OpenSpec 验证通过后同步受管代码并交付单阶段手动命令；本轮不启动完整任务。
