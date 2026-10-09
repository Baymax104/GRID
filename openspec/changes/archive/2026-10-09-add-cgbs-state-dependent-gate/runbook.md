# CGBS 状态门控 E：第一轮训练交付

后续状态：用户已完成该训练；2026-09-17结果见 [training-outcome.md](training-outcome.md)。E 的 NDCG 基本持平 C、对应 Recall 有改善但仍低于 A，未进入正式常量验证。下文保留原交付命令和预定判据，勿因该历史命令重复训练。

## 实现与范围

E 的 arm 为 `content_init_state_gate`。四层各新增三个全零权重，共12参数；原 C 的层级偏置、内容初始化、辅助 CE 到编码器的梯度及训练预算保留。训练和 beam 共用状态评分函数，输入为停止梯度的合法分支 P/Q 归一化熵与归一化 JS。零初始化对齐 C 的前向和已有参数梯度，不承诺后续训练轨迹相同。

本次先执行一个 E 训练。状态适应性是否能改善推荐仍待验证，不自动扩展调参或实验矩阵。

## 启动命令

在 node1 的 `/data3/weizhenyu/projects/GRID` 仓库根目录，由用户手动执行：

```bash
bash ./tiger_catalog_grounded_mechanism_train.sh \
  --data-dir data/beauty \
  --condition e \
  --notes "CGBS state gate; 12 zero-initialized parameters; original auxiliary gradients; same content initialization and 20k budget as C"
```

脚本固定物理 GPU0/1、两进程；默认 seed42、每卡 batch128、20k steps、每500步验证，以 `val/ndcg@10` 选择 checkpoint。默认 SID 来源为 `wandb://4vyi4o6w`，embedding 为 `wandb://3jtt9mpa`，沿用 A/C。E 记录 `mechanism_revision=state-gate-v1`、A reference `5g3wpbg7` 与 C reference `k6jvoo2v`；实际 resolved config 和 artifact lineage 仍须在训练后核验。

该命令不带 `--dry-run`，会正式训练。若用户自行进行 smoke run，可显式追加 `--dry-run`；该结果不能代替正式20k实验。用户 Hydra override 始终具有最终覆盖权，因此改变预算或数据后不能再直接称为与 A/C 同条件。

## 观测与下一道判据

- W&B 检查模型 arm、gate mode、数据 artifact、GPU/预算及完成状态。
- 除原 generation/content loss，记录各层 `gate/layer_{0..3}/alpha_mean`、`alpha_min`、`alpha_max`；这些是有效 teacher 状态的聚合，不是 beam 分布统计。日志变化仅用于排查门控是否退化为常量。
- 先比较同口径的 A/C/E best validation NDCG@10 及对应 Hit@10。不用 E 的训练 loss 或某层前缀改善代替最终推荐收益。
- 未超过 C：本轮参数化未获支持，不自动新增 MLP、特征或 seed；超过 C 但低于 A：仅有局部修正信号；超过 A/C 且 Hit 不下降：再做同 checkpoint 配对复算和动态/常量验证。
- 当前为单 seed 筛选，不能据此宣称论文主方法已经成立。具体证据口径沿用相邻 `gate-decision.md`。

## 已提供的推理接口

根目录 `tiger_catalog_grounded_inference.sh` 选择 E 时自动装配 gate inference model。相同 E checkpoint 支持以下 Hydra override；本轮无需提前执行：

| 模式 | override | 解释 |
| --- | --- | --- |
| dynamic | `model.root.gate_mode=dynamic` | 默认，使用每个状态的动态项 |
| base | `model.root.gate_mode=base` | 仅用该 E checkpoint 学到的层级偏置，属于敏感性检查 |
| fixed | `model.root.gate_mode=fixed`，并显式传入 `model.root.gate_constant_alphas` 与 `model.root.gate_constant_source` | 每层一个合法常量并声明来源；不能把手填列表当作已校准参考 |

base/fixed 仅允许推理，不能用于训练；第一版 E 只接受 `content_scoring=trained/off`，off 不能同时启用常量干预。E 的 mechanism inference 只接受 `--stage screen`（dynamic trained/off）；常量干预通过根目录 inference 脚本显式配置。旧 C/D 的信号实验不受此 E 限制影响。

本轮没有实现或执行正式训练样本均值校准 runner，也没有生成常量。训练记录的确定性抽样身份、上限4096及来源清单协议已在 `design.md` 决策8锁定；待 E 通过第一关后，通过统一入口另行实现和运行。base 和均值常量都不是“最优常量”对照。

## 验证与交付边界

聚焦 CPU 测试覆盖：零初始化等价及 RNG、辅助编码器梯度、非零门控梯度、极端概率/权重、单合法分支、beam/teacher 一致、标签与 batch 划分无关、checkpoint 身份、常量来源校验、非等长 batch 统计、Hydra compose/实例化及 shell 参数/语法。跨 rank 的均值与极值沿用 torchmetrics 的 sum/count/min/max 归约；没有执行真实 GPU/DDP 端到端验证。

交付前按 `./mutagen_sync.ps1` 执行 flush，并确认三个 session 为 `Watching for changes` 且无 conflict。同步仅覆盖项目规定的代码边界；不安装远端依赖或启动训练。

2026-09-17 已完成：两个聚焦测试文件189项通过；随后补充实际单合法分支的 conditional backward 测试1项通过，合计190项。Ruff check/format、shell语法（含四个改动脚本）及 `openspec validate add-cgbs-state-dependent-gate --strict` 通过，`git diff --check` 无空白错误。测试出现4条依赖弃用警告，无测试失败。

同日 Mutagen flush 成功，随后完整 status 确认 `grid-src`、`grid-configs`、`grid-scripts`、`grid-root-code` 均为 One Way Replica、端点连接正常、`Watching for changes`，无 conflict。未启动正式训练，未创建 Git commit。
