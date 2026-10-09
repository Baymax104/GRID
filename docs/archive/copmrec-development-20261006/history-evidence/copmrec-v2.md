# CoPMRec v2：尺度受约束的decoder相关性

2026-10-04用户决定停止v2迭代，开始独立基于真实v0的v3问题分析。v2保留负向testing证据与已有入口，不追加v2.x、权重扫描或续训。当前阶段见 [v3收益转化卡点分析](copmrec-v3-conversion-bottlenecks.md)，v3模型与loss尚未定案。

最新效果：用户推理run [1gu2dsa1](https://wandb.ai/baymaxam/GRID/runs/1gu2dsa1) 已独立核验，Beauty/seed42 testing Recall10=0.069087、NDCG10=0.033285，低于真实v0及LIGER dense；当前实现不晋级。同checkpoint/同候选相关性重排对NDCG为明确负项，完整证据与边界见 [结果报告](copmrec-v2-testing-1gu2dsa1-result.md)。下方“收益未知/待testing”是实现交付时历史状态。

日期：2026-10-04。用户明确v1迭代结束，v2与v1平级、独立基于v0，后续迭代为v2.x。本版采用固定loss lambda=0.01、分数beta=0.5，完成实现后由用户手动开始完整训练；这些常数是用户指定，不是新校准或最优效果结论。

用户已登记v2训练run [beawrjef](https://wandb.ai/baymaxam/GRID/runs/beawrjef)。2026-10-04 13:27:41（Asia/Shanghai）W&B状态finished，最后记录global_step49999；实际配置已确认从零、全evaluation/testing、500验证间隔及固定lambda0.01/beta0.5。完整配置快照见 [run-snapshot.json](evidence/copmrec-v2-beawrjef-20261004/run-snapshot.json)。当前只确认训练身份和配置，best checkpoint与独立testing尚待核验；下方实现阶段证据按当时状态保留。

后续已核验validation-selected best为step42500，NDCG@10=0.04209396615624428；Artifact producer、checkpoint选点记录和本地文件摘要一致。固定引用的完整 [testing推理命令](copmrec-v2-beawrjef-testing.md) 已通过Bash替身/Hydra验证，尚未执行推理或核验testing效果。

## 模型与评分

`ScaledRelevanceCoPMRec`直接继承v0的`JointMixtureLiger`，不继承v1推荐组件。沿用共享encoder、内容投影、SID decoder与mass混合参数，所有推荐参数由同一个AdamW持续训练。

```text
c_ui = cosine(h_u,v_i) / temperature
a_ui = Linear(128,1)(GELU(Linear(d,128)(LayerNorm(d_ui))))
sigma_u = stop_gradient(sqrt(population_variance(c_u over natural_candidates) + epsilon²))
r_ui = 0.5 * sigma_u * tanh(a_ui)
score_ui = c_ui + r_ui
```

生产d=128、temperature=0.07、epsilon=0.001。head只输入完整候选SID的decoder末状态：同一decoder读取`[start,完整SID]`后，取最终block的FFN与残差、final norm/dropout后的末位置hidden state，不取logits或cross-attention中间输出。输出层零初始化，初始r=0；首步head上游排序梯度为0，基础目标仍训练主干，head输出层开始学习后，上游排序梯度随之进入decoder/encoder。生产head共16897参数。

自然集合为当前beam20与全部cold商品的去重并集。训练排序另加入content Top20与真实目标，但这些额外候选不改变sigma统计集合。0/1个自然候选的方差按0处理，sigma为epsilon；尺度用float32计算、仅该统计量detach，不冻结推荐模块。跨用户候选对统一分片256，sigma在完整自然集合上计算，不按chunk分别归一化。训练和推理共享同一评分实现。

保证`abs(r_ui) <= 0.5*sigma_u`，限制数值幅度；不保证与v0排名相同，分差较小的商品仍可翻转。相关性只能重排候选，不能直接补回自然候选之外的商品；beam离散选择没有直接排序梯度。tanh饱和与共享任务梯度冲突仍可能影响收益。

## 联合目标与诊断

```text
loss = SID_CE + content_CE + mixed_NLL + 0.01 * fused_ranking_CE
```

基础三项权重为1，每rank每微批均匀抽4个training用户，以扩展去重候选上的最终score做CE，正例只在训练注入。不新增独立相关性CE、尺度惩罚或自适应权重；beta控制前向修正幅度，lambda控制排序目标在反向中的权重，二者不混用。

training日志新增：

- `ranking_loss`：raw融合CE。
- `weighted_ranking_loss`：0.01乘以raw CE。
- `content_scale`：自然候选sigma的均值。
- `relevance_abs_mean`：训练候选修正绝对值均值。
- `relevance_std_ratio`：自然候选修正标准差/sigma的逐用户均值，应不超过beta（数值容差除外）。
- `relevance_saturation`：训练候选`abs(tanh(a))>0.95`的比例。

推理trace另保存同候选content-only目标排名、sigma、最终TopK的content/relevance分量，便于检查贡献与幅度上界；这些记录依赖标签做诊断，推荐输出评分不读取标签。

## 版本与评价契约

v0就是BMX-116已有的基础方法，当前入口已恢复真实配置；v1迭代在v1.1结束，v1.2已撤回。v2模型/experiment/根入口独立，W&B记录`copmrec_version=v2`。checkpoint严格保存head/score、beta/lambda/epsilon、temperature、chunk、catalog及datamodule提供的评价契约，不能直接恢复v1/v1.1；默认从零，可选v0 weights-only初始化后全参数共同训练。本次命令不启用预训练。

按用户确认，v2与真实v0统一采用FileDataModule：训练读取完整training，验证读取全evaluation，预测读取testing，不再使用selection/audit。v2验证每500微批，按最终融合hybrid NDCG@10选best；v0保留原dense选点。保持seed42、50k更新、train/val/predict每卡batch128/32/32、累积1、双卡有效训练batch256、FP32。配置完整对比见 [v0真实配置恢复与v2对齐](copmrec-v0-config-unification.md)。

已读取 [BMX-116对应run 7y54j4m6](https://wandb.ai/baymaxam/GRID/runs/7y54j4m6) 的实际W&B配置：`val_check_interval=500`、`check_val_every_n_epoch=null`、`accumulate_grad_batches=1`、`log_every_n_steps=50`；首两个验证记录的`trainer/global_step`为499、999。v2通过独立`configs/trainer/copmrec_v2.yaml`设为500，双卡不会再除以卡数。默认累积1时50k更新对应100次验证，预计记录步为499、999、1499直到49999。保持共享MetricCallback和原步数语义；统一用户集合和记录轮次后，v0 dense验证与v2 hybrid融合验证仍是不同评分机制，应在相同testing上比较最终效果。

当前证据只说明实现与契约有效，最终Recall/NDCG未知。既有有界校准阶段2000更新已耗尽，本版不会重置或转移该预算；agent新增完整训练run/实验预算0，未启动或停止已有训练。后续用户手动运行以真实v0作为效果对照，同候选content-only作为机制检查，全evaluation选点后在testing独立评价；匹配收益正向才保留，负向不晋级该实现，口径不匹配/未运行则未知，不自动扩展实验。

## 从零双卡训练命令

在node1仓库根目录通过Bash执行。沿用物理GPU2、3，对应逻辑devices[0,1]；若这些GPU仍被已有训练占用，使用空闲的两个物理编号。使用独立端口29546。

```bash
cd /data3/weizhenyu/projects/GRID
export PATH="$HOME/.local/bin:$PATH"

CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 bash copmrec_v2_train.sh \
  --dataset beauty \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --devices '[0,1]' \
  --seed 42 \
  --master-port 29546 \
  --group copmrec_v2_scaled_decoder_scratch \
  --notes 'CoPMRec v2：基于真实v0，d_ui相关性head，自然候选尺度约束，固定lambda=0.01、beta=0.5，全参数从零联合训练，full evaluation/testing，每500步验证，50k更新' \
  model.root.ranking_loss_weight=0.01 \
  model.root.relevance_beta=0.5 \
  model.root.relevance_scale_epsilon=0.001 \
  model.root.candidate_chunk_size=256 \
  trainer.root.val_check_interval=500 \
  ckpt_path=null \
  pretrained_checkpoint_path=null
```

两个checkpoint引用均null，全部推荐参数随机初始化。根脚本通过`uv run torchrun --nproc_per_node=2 -m src.main experiment=copmrec_v2_train`装配，默认不dry-run，保留notes两种形式、quoting和用户override优先级；只有明确传入`--dry-run`才执行有界smoke。正式run沿用运行字节源码快照、resolved配置及上游lineage记录。

## 推理与验证

新增`copmrec_v2_inference.sh`及同名experiment；训练后固定实际发布的validation-selected best checkpoint引用，再执行本版本推理。当前beawrjef已finished，其best引用核验为`wandb://baymaxam/GRID/beawrjef?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=042500.ckpt`，不能使用v1/v1.1或last代替。恢复v2时beta/lambda/epsilon/temperature/chunk与历史契约必须匹配。

本地98项聚焦检查通过，1项Linux双进程检查在Windows跳过。覆盖独立v0基座、可选预训练与零head、基础三项一致、固定融合CE、自然尺度和正例注入独立、population variance与空/单候选、完整SID/d-only/user映射、候选分片数值与梯度、修正上界、连续联合更新及cross-attention梯度、checkpoint/optimizer/历史恢复、标签独立、trace与脚本配置。Ruff check/format及OpenSpec strict通过。9个v1/v1.1模型/配置/入口与撤回后的指纹完全一致。

node1的20项CPU检查通过，包含两进程Gloo连续联合更新、梯度有限性和参数同步。文档中的完整训练命令及推理入口通过uv替身与Hydra配置核验，确认双卡、50k更新、固定lambda/beta及两个checkpoint引用为null，没有实际启动训练。18个运行文件的本地/远端指纹一致，9个v1/v1.1运行文件保持原字节；Mutagen三个session均Watching、无conflict。证据见 [verification.json](evidence/copmrec-v2-20261004/verification.json)。最小CPU单元更新不构成推荐收益，也未验证完整GPU DDP吞吐和显存。

上段为初次交付核验。验证间隔对齐后，11项启动脚本/配置检查通过，Ruff与OpenSpec strict通过；node1默认组合与文档命令替身均确认验证间隔500，19个运行文件指纹一致，v1/v1.1仍为2500、原运行字节一致，Mutagen三个session均Watching且无conflict。补充证据见 [validation-schedule-verification.json](evidence/copmrec-v2-20261004/validation-schedule-verification.json)，不会覆盖原始证据。已经启动的训练进程不会自动读取此次配置修改，后续新启动的v2采用500间隔；本次没有启动、停止或恢复训练。

以上是统一真实v0之前的分阶段证据。随后按用户要求完成全配置统一，当前评价数据为全evaluation/testing；最新62项聚焦检查、resolved配置对比及远端核验见 [统一记录](copmrec-v0-config-unification.md)。旧selection/audit checkpoint恢复到当前full协议会因已有history契约不匹配而拒绝，本次从零命令仍明确两个checkpoint引用为null。
