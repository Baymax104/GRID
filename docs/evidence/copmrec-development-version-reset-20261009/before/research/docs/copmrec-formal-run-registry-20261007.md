# 正式 W&B Run Registry｜CoPMRec v5.3

2026-10-08 BMX-116正式主矩阵已核验完成：9/9新CoPMRec训练、9/9Testing，9/9原LIGER hybrid配对输出独立核对，原baseline保持复用。已完成checkpoint lineage、testing用户/标签、合法唯一Top10、CoPMRec历史排除与四指标独立复算；九个配对bootstrap及三seed均值/标准差已归档。原LIGER历史政策与CoPMRec不同，保留完整方法比较边界。详见GRID docs/evidence/copmrec-main-completion-20261008/；dense内部对照与消融不计本父任务完成。

| Dataset | CoPMRec NDCG@10 mean +/- std | LIGER hybrid NDCG@10 | Gain | CoPMRec Recall@10 mean +/- std | LIGER hybrid Recall@10 | Gain |
| -- | -- | -- | -- | -- | -- | -- |
| beauty | 0.04355895 +/- 0.00109371 | 0.02992889 | 45.54% | 0.07583956 +/- 0.00269898 | 0.05570213 | 36.15% |
| sports | 0.02398604 +/- 0.00054744 | 0.01574162 | 52.37% | 0.04356986 +/- 0.00067478 | 0.02864393 | 52.11% |
| toys | 0.04668824 +/- 0.00031526 | 0.02629864 | 77.53% | 0.07699705 +/- 0.00066238 | 0.04837214 | 59.18% |


2026-10-08 seed2026三个训练已finished并通过固定50k与best审计，Testing命令已补齐，未启动推理。累计训练完成9/9、Testing执行完成6/9；seed2026三个Testing待执行，已有六个Testing独立核验待完成。回执：GRID docs/evidence/copmrec-seed2026-audit-20261008/。

2026-10-08 六个seed42/200正式Testing全部finished、退出码0，运行中0；用户授权的六条test命令已执行完毕。训练完成6/9、Testing执行完成6/9；输出与用户集合、指标独立复算待核验，完整配对暂不计Done。seed2026未启动。回执：GRID docs/evidence/copmrec-testing-launch-20261008/。

2026-10-08按用户授权启动六个单卡Testing，使用各自审计best v0文件/SHA256，统一group、testing split和独立tmux。6个任务均已验证checkpoint加载、local GPU映射和runtime源码快照。当前W&B running 1 / finished 5，完整单元有效性与独立复算待完成；seed2026未启动。

| Issue | Dataset | Seed | GPU | tmux | Testing run | State |
| -- | -- | -- | -- | -- | -- | -- |
| BMX-120 | beauty | 42 | 7 | copmrec_test_beauty_s42_20261008 | vosmuihm | finished |
| BMX-119 | sports | 42 | 1 | copmrec_test_sports_s42_20261008 | tl55l87o | finished |
| BMX-17 | toys | 42 | 3 | copmrec_test_toys_s42_20261008 | pxh4ffi6 | finished |
| BMX-121 | beauty | 200 | 5 | copmrec_test_beauty_s200_20261008 | 95rx50ma | finished |
| BMX-15 | sports | 200 | 6 | copmrec_test_sports_s200_20261008 | 5dx507jd | running |
| BMX-18 | toys | 200 | 0 | copmrec_test_toys_s200_20261008 | jlod7okq | finished |


最新训练审计：seed42/200共6个训练finished并通过固定50k及best核对，running 0；Testing尚未启动，seed2026尚未启动。六个独立Testing命令已填入各子issue的精确best v0 URI及SHA256并验证脚本参数/Hydra配置。详情见GRID docs/evidence/copmrec-training-audit-20261007/。此前运行快照保留作历史。

2026-10-07 最新运行状态：seed42三个训练均finished、退出码0，best审计与Testing待完成；本轮按用户明确授权启动seed200三个双卡tmux训练，并允许共享显存足够的GPU。累计训练started 6 / finished 3 / running 3，Testing启动与完成仍0；seed2026待后续启动。启动回执：GRID docs/evidence/copmrec-tmux-seed200-20261007/。

2026-10-07按用户纠正：正式baseline为LIGER hybrid，原9组正式训练/Testing结果可以复用。BMX-132与BMX-133～141恢复原始标题、描述、Done及有效/基线标签；不要求重新训练、独立Validation或重复Testing。CoPMRec开发run仍只用于历史，不完成新正式CoPMRec任务。

## 已完成 LIGER 正式基线

| Dataset | Seed | LIGER issue | Train run | Testing run | Best step | NDCG@10 | Recall@10 |
|---|---:|---|---|---|---:|---:|---:|
| Beauty | 42 | [BMX-133](https://linear.app/baymax104/issue/BMX-133) | `35ig0tz6` | `042139al` | 45000 | 0.0298675 | 0.0559406 |
| Beauty | 200 | [BMX-134](https://linear.app/baymax104/issue/BMX-134) | `20qs5i9w` | `0b2g9ktg` | 45000 | 0.0297251 | 0.0559853 |
| Beauty | 2026 | [BMX-135](https://linear.app/baymax104/issue/BMX-135) | `xeg0lbww` | `eg0morwo` | 47500 | 0.0301941 | 0.0551804 |
| Sports | 42 | [BMX-136](https://linear.app/baymax104/issue/BMX-136) | `x69dhrk6` | `p81iez3h` | 43000 | 0.0155035092 | 0.0282319231 |
| Sports | 200 | [BMX-137](https://linear.app/baymax104/issue/BMX-137) | `wnpv48oe` | `gppl7usj` | 50000 | 0.01550899795354818 | 0.028316197539187595 |
| Sports | 2026 | [BMX-138](https://linear.app/baymax104/issue/BMX-138) | `9chlbjns` | `oqdjukup` | 49500 | 0.016212353065392596 | 0.029383673240069668 |
| Toys | 42 | [BMX-139](https://linear.app/baymax104/issue/BMX-139) | `l99i8w61` | `5vd0g7ak` | 46500 | 0.02580209533967576 | 0.04729033587471667 |
| Toys | 200 | [BMX-140](https://linear.app/baymax104/issue/BMX-140) | `7lbsl22u` | `b591uoen` | 45000 | 0.0268886263 | 0.0492994024 |
| Toys | 2026 | [BMX-141](https://linear.app/baymax104/issue/BMX-141) | `rsz30j8q` | `8hjgn6b5` | 46500 | 0.0262051905 | 0.0485266845 |

以上18个run已只读在线核对，均finished，Testing为original hybrid、生成20；原best Artifact可访问，指标与原issue一致。原训练配置为双卡50k/global256/FP32，原Validation在train run内选点，没有独立Validation run要求。具体checkpoint URI与原命令保留于各恢复的issue和research-state.yaml；未重新核实的文件SHA256继续为空。

原Testing使用src.recommendation.liger.Liger，不把新增HistoryExcludedHybridLiger写成既有结果的协议。新主表接入前仍核split/keys、输入与历史资格；必要重评分复用原best并另列成本，不重置已完成基线或自动重训。

## 新 CoPMRec 正式矩阵

2026-10-07用户授权首批三个tmux双卡训练并允许共享GPU。Beauty42/BMX-120：gshpyn49（GPU1,5），Sports42/BMX-119：jk4zk19n（GPU3,6），Toys42/BMX-17：hs72xkan（GPU0,7）；三单元与父BMX-116为In Progress。训练启动3/9、完成0/9，Testing、best与metrics仍null，其余六单元未启动。训练内validation选best，审计后直接Testing，不额外Validation；validation_run=null标记not_required，开发run禁止复用。

主配对新增仅9训练/450k更新+0额外Validation+9Testing。既有LIGER9组结果复用；dense内部另9Testing，使用原对应LIGER best、0训练/0Validation，命令待准备，不替代hybrid主表。CoPMRec issue已按其他实验的六段模板统一，只有训练与Testing两条运行命令。

## 身份与证据边界

* 新CoPMRec：evidence_phase=formal、formal_release_id=copmrec-v5.3、copmrec-formal-v53 tag，checkpoint含正式来源契约。
* 主实验group：paper_main_copmrec_beauty、paper_main_copmrec_sports、paper_main_copmrec_toys；与其他方法的paper_main_<method>_<dataset>一致。同数据集各seed、train/Testing共用group，以config/notes/tags区分版本及阶段。
* 复用LIGER：保留原run/config/notes与已完成证据，不伪造其拥有后来新增的formal_release元数据。
* dense内部：evidence_phase=internal、copmrec-internal-v53 tag，只进入内部表。
* TIGER/LETTER/SASRec原有效状态保留；所有既有baseline接入新主表前核实际协议。
* 恢复快照、18个run的当前config/summary/Artifact与Linear逐字回读回执：GRID docs/evidence/liger-baseline-restore-20261007/。
* 首批三个训练由用户明确授权启动；授权不延伸到其余seed、Testing或消融。tmux、GPU与实际配置/源码快照回执见GRID docs/evidence/copmrec-tmux-train-launch-20261007/。

## 此前开发清理回执（保留原记录）

下文的旧范围与计数按各次删除事件追溯；“新正式不使用开发run”针对开发CoPMRec及开发v4/budget对照，不再排除原有效LIGER正式主矩阵。

## 2026-10-07 后续追加：v4／LIGER budget 开发对照清理

按用户追加说明，3个CoPMRec v4阶段的LIGER开发对照与6个LIGER budget训练/评价run也属于开发阶段，已全部删除；本轮关联19个普通Artifact版本已不可访问，3个系统history版本仍保留并单列。新增44项效果已保存至GRID历史版本文档，回执位于`docs/archive/copmrec-development-20261006/wandb-delete-v4-budget-20261007/`。此前“保留27个LIGER comparator”描述旧范围；目前保留原hybrid主矩阵18个历史run及9个固定上游。新正式实验仍不以开发run计完成。

---

## 2026-10-07 开发 W&B 删除结果

用户追加删除授权后，已删除82个CoPMRec开发run与196个普通用户Artifact版本；1个登记run ID原已不存在。27个历史LIGER comparator和9个固定上游run保留且已核验。开发效果已完整落入GRID的`docs/archive/copmrec-development-20261006/versions.md`：82个现存run的1138项Recall/NDCG、差值和区间原值，另存完整config/summary/Artifact metadata及35个运行的Validation NDCG曲线。

57个系统管理的wandb-history/events版本，W&B接口拒绝单独删除，所属run删除后仍可查询；它们列为服务侧待清理，不计入196个已删除普通版本。完整回执位于`wandb-delete-20261007/deletion-receipt.json`及`verification.json`。下文“保留Artifact”等语句描述之前的2026-10-06归档阶段；当前开发W&B链接已失效，使用本地历史快照追溯。新正式矩阵仍不使用任何开发run/CP作为完成来源。

---
