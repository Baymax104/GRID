## ADDED Requirements

### Requirement: 单次50k整体训练

系统 MUST 支持推荐模型随机初始化后全部模块从step0联合优化，固定生产总预算50000 optimizer updates、global256、FP32、AdamW主干peak.0003/itempeak.002、wd.035、warm2500及同一cosine50000/min0；MUST 禁用外部pretrained checkpoint、teacher、阶段optimizer重建和checkpoint pooling。

#### Scenario: 正式从头运行
- **WHEN** 用户从仓库根运行统一训练入口
- **THEN** ckpt_path与pretraining loader均为null，T5/projection/gate随机或规定零初始化、残差零表，单一训练进程链到50000，源/终端/实际更新证据可审计。

#### Scenario: 拒绝旧权重和超预算
- **WHEN** 输入v0/v4 warm权重、错误契约、超过50k的checkpoint或finishedrun再次fit
- **THEN** 模型或driver拒绝继续，不能将其标为scratch50k或重置累计成本。

#### Scenario: 故障完整状态恢复
- **WHEN** 同一未结束job需故障恢复
- **THEN** 仅恢复其完整own optimizer/scheduler/globalstep且不重启日程，driver绑定原run并记录真实消费；weights-only恢复不能通过此条件。

### Requirement: 单模型共享残差与概率联合目标

模型 SHALL 复用v0 unit SID CE、unit full-catalog content CE、unit合法前缀mass mixture NLL与learnedalpha；seen商品残差同时服务历史与目录、scale1/cold0；MUST 不加固定alpha、mixed终排、scorebias或训练historyCE改动。

#### Scenario: 初始等价与真实学习
- **WHEN** 固定同seed构造v0与新模型且新表为零
- **THEN** 基座RNG及step0三loss数值等价，三项各自向seen残差提供有限梯度，cold梯度为0。

### Requirement: 原生checkpoint选择与共同资格部署

训练 SHALL 每500更新按原完整catalog raw dense ValN10选ownbest；独立完整Val/Test SHALL 单卡应用同LIGER输入有效完整SID历史排除、cold可选、稳定catalog row tie、完整目录dense打分，MUST 不让label/userkey进入排名。

#### Scenario: 原生验证口径
- **WHEN** Trainer验证以选择checkpoint
- **THEN** rawdense选择口径与原native一致，标签只参与loss/metric，不改变排名；history部署改变不伪装训练收益。

#### Scenario: 单卡完整部署
- **WHEN** 统一predict读取ownbest
- **THEN** 输出keys/predictions bundle、合法唯一Top10、零有效history，world_size必须1，单模型无外部成员。

### Requirement: 固定预算效果与成对复现

研究 MUST 固定stage累计上限3正式训练/150k新增更新、3完整Val及3新Test，不重置旧预算。每个模型从0到50k；seed42先Val晋级，随后冻结同方法并训练native43/candidate43，再仅用预选checkpoint的单卡Testing验收。原50k samepolicy native42为主分母，64kpool仅辅助。两个seed SHALL 各自双R10/N10相对同seed native≥8%且paired绝对CI95下界正，不能靠均值或较好seed代替。

#### Scenario: 首个候选完整Val晋级
- **WHEN** seed42 fullVal相对wdms8w77达到R10≥0.10470151589679383且N10≥0.0583728104417629、两paired绝对CI下界正
- **THEN** 冻结方法并消费其余两训练用于native43/candidate43的成对复现，不以Test挑方法。

#### Scenario: 未通过晋级或复现
- **WHEN** 任一固定阶段不满足预设门槛
- **THEN** 如实记录正负／未确认结果、消费与剩余额度，不声称已达目标、不扫阈值/超参/seed；完整目标仍由goal状态保留，后续决定须有新实际证据。

### Requirement: 可复现入口与真实来源

系统 MUST 使用src.main/Hydra及根脚本，支持显式dry-run、两种notes形式和末尾override；多卡torchrun、单卡predict、字段级上游引用、公共writers和source snapshot保持现有契约。正式任务 SHALL 通过官方Mutagen三session flush后核actual运行bytes、CP/Artifact producer/digest、原始split/labels/keys与独立metrics。

#### Scenario: 从归档重跑
- **WHEN** 从固定命令、resolved config、runtime archive及上游SID/content身份重新运行
- **THEN** 不依赖旧训练checkpoint或阶段选择，能构造相同50k整体模型与单成员评价链，报告实际seed、预算、单数据集和历史provenance边界。
