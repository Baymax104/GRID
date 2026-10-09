# BRIR 实施与手动运行

日期：2026-09-15。用户已采纳方向；代码实现及轻量验收完成，真实 GPU 效果、峰值显存和速度尚未验证。旧 MIR/CGBS checkpoint 不兼容 BRIR masked-mean/raw-content 基础评分契约。

## 1. 本轮实现

- 四种 arm：`base`、`dense`、`prefix_free`、`brir`。后两组冻结基础模型，dense 组保留冻结候选来源副本。
- 所有组使用相同 SID 历史和 keyed 内容。基础 full-softmax；三分支使用共同基础 top64 非正例及16个历史hash确定的随机非正例。优化器独立初始化。
- 训练使用0、1双卡DDP，每卡microbatch8 × 累积8 × 2卡 = 有效batch128；标定、推理与审计单卡。基础20k优化步，分支各5k。每1000优化步验证，保留全部验证点 checkpoint；best与last分别发布，基础分支来源必须用20k last。
- 标定在training split按用户key hash选择1024条最后训练历史；10/128分数间距的 midpoint median/2 决定delta。该抽样没有对32条增强训练前缀作均匀抽样。
- `reference`、`dynamic`、`fixed`检索；普通推理保留搜索上界和成本辅助产物，audit额外核对全目录reference与固定64/128/256。
- 原始bound、exact集合、exact顺序及verified分别记录；数值余量为经验防护，不是浮点严格证明。Top-K排名与真实推荐效用分开解释。

## 2. 现在只启动共享基础训练

从node1的 `/data3/weizhenyu/projects/GRID` 根目录运行：

```bash
NPROC_PER_NODE=2 bash ./brir_train.sh \
  --data-dir data/beauty \
  --dataset beauty \
  --group rkmeans \
  --semantic-id-path wandb://4vyi4o6w \
  --embedding-path wandb://3jtt9mpa \
  --arm base \
  --gpu 0,1 \
  --seed 42 \
  --notes "BRIR shared dense base; Beauty seed42; full item objective; fixed 20k-step source for matched branches"
```

SID/embedding引用与现有Beauty取证脚本一致，实际Artifact由共享解析器记录lineage。末尾加 `--print-only` 仅打印，无W&B访问或模型执行；`--dry-run` 会执行统一入口的小运行，不能计入正式结果。初次实现不自动启动GPU smoke。

完成后先核对run状态finished、`brir_arm=base`、`experiment_protocol=brir-v1`、global_step=20000，以及checkpoint_last Artifact。不要用best初始化三分支。基础run预计名称为 `brir_base_beauty_train_seed42`，真实ID由W&B分配。

## 3. 基础完成后的标定

以下 `BASE_RUN_ID`、`CALIBRATION_RUN_ID`、`BRANCH_RUN_ID` 是待替换占位符，当前没有这些新run。

```bash
bash ./brir_calibration.sh \
  --data-dir data/beauty --dataset beauty --group rkmeans \
  --semantic-id-path wandb://4vyi4o6w --embedding-path wandb://3jtt9mpa \
  --checkpoint-path 'wandb://BASE_RUN_ID?role=checkpoint_last' \
  --arm base --gpu 0 --seed 42 \
  --notes "BRIR delta calibration; 1024 key-hash training histories; frozen fixed-step base"
```

检查 `brir_calibration` Artifact是否完整1024行、training来源、基础权重fingerprint一致及delta非退化。零间距会使分支入口失败，不会自动扩大delta。

## 4. 三条分支分别启动

下面命令以 `brir` 为例；分别将 `--arm` 改为 `dense`、`prefix_free` 和 `brir` 执行三次。每次使用同一个base与calibration引用，不从另一分支继续训练。三次分支各5k优化步，不需要再次训练20k基础。

```bash
NPROC_PER_NODE=2 bash ./brir_train.sh \
  --data-dir data/beauty --dataset beauty --group rkmeans \
  --semantic-id-path wandb://4vyi4o6w --embedding-path wandb://3jtt9mpa \
  --base-checkpoint-path 'wandb://BASE_RUN_ID?role=checkpoint_last' \
  --calibration-path 'wandb://CALIBRATION_RUN_ID?role=brir_calibration' \
  --arm brir --gpu 0,1 --seed 42 \
  --notes "BRIR matched Beauty branch; common frozen base and negatives; 5k optimizer steps"
```

分支初始化用 `--base-checkpoint-path`；同一分支的中断恢复才用 `--checkpoint-path`。初始化和恢复都检查checkpoint契约，不能混用旧MIR/CGBS或错目录、错温度、错序列长度、错seed/基础步数的文件。

## 5. 冻结分支的搜索审计及全量输出

```bash
bash ./brir_audit.sh \
  --data-dir data/beauty --dataset beauty --group rkmeans \
  --semantic-id-path wandb://4vyi4o6w --embedding-path wandb://3jtt9mpa \
  --base-checkpoint-path 'wandb://BASE_RUN_ID?role=checkpoint_last' \
  --calibration-path 'wandb://CALIBRATION_RUN_ID?role=brir_calibration' \
  --checkpoint-path 'wandb://BRANCH_RUN_ID?role=checkpoint_last' \
  --arm brir --gpu 0 --seed 42 \
  --notes "BRIR frozen score-bound audit; paired128; reference dynamic fixed64 fixed128 fixed256"
```

audit只取同样hash的128用户，输出 `brir_audit.pt`，不能当完整推荐质量实验。全量evaluation输出使用 `brir_inference.sh` 和同样参数；默认dynamic，`--policy reference`可得到全目录reference的完整输出，固定候选使用 `--policy fixed model.root.candidate_count=128`。基础模型输出用 `--arm base` 和基础checkpoint，省略base/calibration输入。

普通输出仍是标准keyed recommendation bundle，并单独发布 `brir_search.pt`；审计输出含完整目标SID、目标item key、输入hash、各策略Top-K keys/scores/计数/时间及证书字段。计时中的 `base_seconds` 包含编码、基础评分及必要目录向量构造；各策略seconds只计后续检索。目录向量缓存会影响冷/热时间，正式比较需匹配warmup、硬件和计时范围，不能把一次顺序audit计时直接当生产p95。

## 6. 验收与研究门禁

聚焦测试60项通过，包括梯度、冻结候选、缓存失效、checkpoint篡改拒绝、候选边界/平局/预算、标定writer往返、证据校验、全部配置实例化和Git Bash参数解析。Ruff、shell语法与OpenSpec strict验证通过。没有运行真实Trainer实验、训练、推理或GPU矩阵。

开发门禁沿用设计：BRIR的全evaluation NDCG@10相对更强的dense继续训练/无前缀对照至少+1%，Hit@10不下降；动态审计与reference集合和顺序一致，平均精算不超过目录10%，实测时延具有价值，并检查固定候选替代方案。用户级区间不是跨seed证据。Beauty通过后才研究Sports，不自动恢复54-run MIR矩阵。

与实现有关的主要限制：训练支持一或两个进程，标定/推理/审计单进程；冻结候选来源增加dense继续训练的成本；标定与搜索可能退化；GPU峰值显存和端到端吞吐尚未测量。不得静默更改有效batch、验证间隔、sample或checkpoint选择来掩盖运行成本。

双卡调整验收：44项模型/配置/脚本测试通过；未运行真实GPU/DDP训练。单卡累积16，双卡累积8，有效batch与优化步验证间隔一致。
