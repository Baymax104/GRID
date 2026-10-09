# CoPMRec v2 beawrjef：固定best checkpoint的testing命令

后续执行结果：用户run1gu2dsa1已finished并通过独立testing复算，当前v2相对真实v0/LIGER dense为负向，见 [结果报告](copmrec-v2-testing-1gu2dsa1-result.md)。下方命令及“尚未启动”说明为交付时记录，本次审计未重新运行推理。

日期：2026-10-04。训练run为 [beawrjef](https://wandb.ai/baymaxam/GRID/runs/beawrjef)，当前finished。已核验按`val/ndcg@10`、mode=max选出的best checkpoint为`checkpoint_epoch=000_step=042500.ckpt`，内部global_step42500；Artifact producer、selection元信息、checkpoint的ModelCheckpoint记录和node1本地文件MD5摘要一致。对应验证记录global_step42499的NDCG@10=0.04209396615624428、Recall@10=0.08737646788358688；这些是validation指标，不是testing效果。

## 推理命令

从node1仓库根目录用Bash执行，物理GPU2/3对应逻辑[0,1]，端口29547。checkpoint URI的`alias=v0`是不可变Artifact版本编号，模型仍为CoPMRec v2。

```bash
cd /data3/weizhenyu/projects/GRID
export PATH="$HOME/.local/bin:$PATH"

COPMREC_V2_BEST_CKPT='wandb://baymaxam/GRID/beawrjef?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=042500.ckpt'

CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 bash copmrec_v2_inference.sh \
  --dataset beauty \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --checkpoint "$COPMREC_V2_BEST_CKPT" \
  --devices '[0,1]' \
  --seed 42 \
  --master-port 29547 \
  --group copmrec_v2_beawrjef_testing \
  --notes 'CoPMRec v2 testing：beawrjef验证选出的best step42500，full testing，content加d_ui相关性，lambda=0.01、beta=0.5，保留候选trace用于效果评估' \
  model.root.ranking_loss_weight=0.01 \
  model.root.relevance_beta=0.5 \
  model.root.relevance_scale_epsilon=0.001 \
  model.root.candidate_chunk_size=256 \
  pretrained_checkpoint_path=null
```

该入口在完整`data/beauty/testing`上预测，保留user_id键、融合hybrid终排和候选trace，记录`test/recall@5`、`test/recall@10`、`test/ndcg@5`、`test/ndcg@10`及`test/user_count`。默认发布recommendation_output和liger_candidate_trace Artifact，供独立指标检查和同候选content-only贡献分析。没有标签注入推理候选。

## 后续效果评价

此次仅交付命令和核验best引用，没有启动推理。用户运行后，将新inference run作为testing结果来源，核验实际消费的checkpoint/输入Artifact、用户及标签身份、输出合法性，独立复算Recall/NDCG；再与BMX-116真实v0及LIGER dense的匹配Beauty/seed42 testing结果比较。Trace用于区分候选覆盖和融合重排影响，不以训练summary末值或validation收益替代testing收益。本次不新增或重置训练预算。

核验记录见 [best-checkpoint-and-command.json](evidence/copmrec-v2-beawrjef-20261004/best-checkpoint-and-command.json)。
