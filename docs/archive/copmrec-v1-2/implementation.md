# CoPMRec v1.2：decoder表示相关性head

> 历史归档：该版本已按用户要求撤回，以下命令已失效。当前使用v1.1，见同目录README。

日期：2026-10-04。用户授权简化head并记为v1.2。此次交付实现与运行命令，完整训练由用户手动开始，推荐收益尚未验证。

## 结构与训练

v1.2只改变相关性head的输入：

```text
v1.1: [h_u, v_i, h_u*v_i, d_ui] → Linear(512,128) → GELU → Linear(128,1)
v1.2: d_ui                     → Linear(128,128) → GELU → Linear(128,1)
```

生产配置head参数65793→16641，减少约74.7%，不是整个模型参数减少74.7%。输出层继续零初始化，初始残差为0；第一步残差上游梯度为0，基础目标仍训练主干，后续排序梯度可进入decoder/cross-attention/encoder。

`d_ui`取同一T5 decoder读取`[start,完整候选SID]`后的`last_hidden_state[:, -1]`，即最终block的FFN与残差之后，再经过final layer norm及dropout的隐藏状态；不是logits或cross-attention的中间输出。用户条件来自cross-attention读取整个encoder历史，而不是直接拼接h_u。SID量化可能丢失候选连续内容细节，不能声称d与h/v信息完全等价。

```text
score(u,i) = cos(h_u,v_i) / 0.07 + head(d_ui)
loss = SID_CE + content_CE + mixed_NLL + 0.05726763550972437 * ranking_CE
```

校准权重继承当前v1.1以隔离结构变化；它来自旧head的training-only梯度校准，不宣称是v1.2最优权重，也不在推理时乘head输出。内容分数继续使用h/v，简化只影响残差路径。推荐参数全部可训练、单一优化器，从第一步联合训练，不冻结teacher。

继承v1.1跨用户合并候选评分和chunk256；beam20及全部cold构成推理候选，训练另加入content Top20和正例，统一去重。每rank每微批均匀抽4个排序用户，保留固定selection/audit、50k更新及validation NDCG选点。候选覆盖缺失、无界残差和共享梯度冲突仍可能存在。

## 当前证据与决策

通过W&B MCP对baymaxam/GRID的指定run进行只读summary及有界history读取，快照见[wandb-snapshot.json](../../evidence/copmrec-v1-2-20261004/wandb-snapshot.json)。采样history不宣称穷尽。

| 已记录验证点 | 校准v1.1 Recall@10 | 校准v1.1 NDCG@10 | 等权v1.1 Recall@10 | 等权v1.1 NDCG@10 |
|---|---:|---:|---:|---:|
| 22.5k | 0.04847509 | 0.02123770 | 0.01323674 | 0.00679011 |
| 27.5k | 0.04400322 | 0.01891388 | 0.01520436 | 0.00767391 |

校准run57hzkcol快照时仍running，global_step29999，但summary中验证数来自27499而不是29999；此前改善未消除低指标及后续回落。旧v0 run7y54j4m6在27499的全量dense参考为0.08181658/0.04170942，与当前selection hybrid不匹配，不能直接作因果结论。v1.2检验的是“由完整SID的用户条件表示独立提供修正，能否减少内容路径重复并改善最终排序”，不是已经确认的修复。

此前两臂各1000更新只支持降低权重的短程Recall恢复，NDCG配对区间仍跨零，不是ranking vs无ranking的消融。该阶段2000更新已耗尽，不能因本次变更重置；本次新增agent训练预算/运行数0，无自动权重扫描、补对照或audit。完整实验保持用户手动执行，以校准v1.1为结构对照，同协议选点与独立评价支持最终结论。改善则保留候选；匹配结果仍更差则不晋级head；局部波动或口径不匹配则保持未知，不自动扩大预算。

## 从零双卡训练命令

在node1仓库根目录用Bash执行。沿用物理GPU2、3，对应逻辑devices[0,1]；已有训练未被本次操作停止，请在这些GPU可用时执行，或将CUDA_VISIBLE_DEVICES改为可用的两个物理GPU。使用独立端口29545。

```bash
cd /data3/weizhenyu/projects/GRID
export PATH="$HOME/.local/bin:$PATH"

CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 bash copmrec_v1_2_train.sh \
  --dataset beauty \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --devices '[0,1]' \
  --seed 42 \
  --master-port 29545 \
  --group copmrec_v1_2_decoder_head_scratch \
  --notes 'CoPMRec v1.2：head仅使用完整SID的decoder末状态，校准排序权重0.0572676，从零训练，固定selection/audit，50k更新' \
  model.root.candidate_chunk_size=256 \
  model.root.ranking_loss_weight=0.05726763550972437 \
  model.root.allow_ranking_loss_reweighting=false \
  ckpt_path=null \
  pretrained_checkpoint_path=null
```

每卡batch128、累积1、有效batch256，FP32，50k optimizer更新。入口为uv run torchrun -m src.main experiment=copmrec_v1_2_train，默认不dry-run。两个checkpoint字段显式null，推荐参数随机初始化，不恢复旧checkpoint。相同seed不保证不同head结构的dropout与训练轨迹逐位相同。

## 推理及兼容

新增copmrec_v1_2_inference.sh及同名experiment；推理使用本版本训练实际发布的validation-selected best checkpoint，不能拿v1.1或last引用替代。当前未产生v1.2最佳checkpoint，尚无可固定的真实推理引用。v1.2记录copmrec_version、相关性version、head_input、chunk、loss和评价历史契约，拒绝直接恢复旧相关性版本。可选v0 weights-only预训练继承原能力，加载后仍全参数训练；上方命令不启用它。

v0/v1/v1.1保留现有结构、参数名、初始化及恢复规则。旧版本只将head构造与残差计算抽成可覆写方法，实际head和评分不变。

## 验证

本地87项聚焦测试通过、3项Linux Gloo测试跳过。覆盖d-only评分、完整SID、用户对应与候选尾块、内容基础分数、零初始化、加权loss、非零head的cross-attention/encoder/历史内容投影梯度、联合更新、checkpoint/optimizer恢复、跨版本拒绝、标签独立及trace校验；旧版本数值和梯度回归通过。入口核验包含Hydra compose、LF字节、shell语法、notes两种形式、quoting、空值/错误参数、额外override和两进程argv。

node1另外10项检查通过，包括v1/v1.1/v1.2各两进程连续更新、全参数一致及非等长分片指标；其中v1.2结构检查7项、双进程检查1项。文档完整命令使用uv替身，仅捕获argv并compose配置，检查了随机初始化所需的两个null引用、权重、双卡、batch和50k更新；没有启动实际训练。初次远端配置断言误把predict的audit字段当成validation字段，修正后通过；val_dataloader在代码中固定selection，该错误没有触发任何训练或测试。

Ruff check/format、OpenSpec strict通过。Mutagen flush成功，三个会话均Watching且无conflict，9个运行文件SHA256与本地一致。核验源码与结果见 [verification.json](../../evidence/copmrec-v1-2-20261004/verification.json) 及同目录verification-source.txt。CPU最小输入验证不代表完整GPU DDP吞吐、显存或推荐收益。本次未启动/停止完整训练或新建W&B run。
