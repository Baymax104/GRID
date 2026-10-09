# CoPMRec v3 rbha00vx：训练核验与固定best的Testing命令

后续结果：用户推理run [zy8946l2](https://wandb.ai/baymaxam/GRID/runs/zy8946l2) 已完成并通过独立Testing复算，v3主指标@10未建立净收益，首层遗漏改善、后层剪枝仍在，见 [结果报告](copmrec-v3-testing-zy8946l2-result.md)。下方“尚未运行”按命令交付时历史记录保留。

日期：2026-10-04。用户报告训练完成，本次实时核验 [rbha00vx](https://wandb.ai/baymaxam/GRID/runs/rbha00vx) 为finished，Beauty/seed42，双卡、从零、50k更新。实际data/model/trainer/callbacks配置与交付v3的Hydra装配一致：沿用真实v0的训练配置，首层Max、后续Mass，无新增ranking head/loss。配置为全evaluation dense验证；下列数值是该run记录的validation指标，本次没有独立重跑验证。

## best与来源

checkpoint Artifact为 `copmrec_beauty_v3_train-checkpoint:v0`，producer为rbha00vx，selection=best、monitor=val/ndcg@10、mode=max。文件为 `checkpoint_epoch=000_step=043500.ckpt`，内部global_step43500；对应历史记录为trainer/global_step43499。

| 项目 | 值 |
|---|---|
| best validation NDCG@10 | 0.04560798406600952 |
| 对应 validation Recall@10 | 0.08981499820947647 |
| checkpoint alpha | 0.668786346912384 |
| 聚合契约 | [max, mass, mass, mass] |
| 训练SID Artifact digest | 19ace08283163fbe687287fbacaa3842 |
| 训练embedding Artifact digest | ab56af975eac589c27eed6094482cbd7 |

完整读取100个验证记录，最高NDCG与Artifact best_model_score及checkpoint内ModelCheckpoint记录一致。node1文件大小/MD5与Artifact entry一致，保存参数有限、模型版本v3及聚合/目标契约正确。实际源码快照producer为rbha00vx，source_sha256为 `ab9d0d4073a7545ab763e2da739c180c473c8853d16a6bfda0d02a238f2eacd1`；三个源码Artifact文件的摘要和manifest SHA256通过，归档中25个相关运行文件与交付版本hash一致。

## 单卡Testing命令

从node1仓库根目录用Bash执行。物理GPU2映射Trainer逻辑[0]，NPROC_PER_NODE=1；端口29548可按占用修改。URI中的alias=v0是不可变Artifact版本编号，模型为CoPMRec v3。

```bash
cd /data3/weizhenyu/projects/GRID
export PATH="$HOME/.local/bin:$PATH"

COPMREC_V3_BEST_CKPT='wandb://baymaxam/GRID/rbha00vx?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=043500.ckpt'

CUDA_VISIBLE_DEVICES=2 NPROC_PER_NODE=1 bash copmrec_v3_inference.sh \
  --dataset beauty \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --checkpoint "$COPMREC_V3_BEST_CKPT" \
  --devices '[0]' \
  --seed 42 \
  --master-port 29548 \
  --group copmrec_v3_rbha00vx_testing \
  --notes 'CoPMRec v3 testing：rbha00vx验证选出的best step43500，full testing，首层Max后续Mass，beam20加全部cold、content终排；保留candidate与path trace用于同v0候选损益分析' \
  callbacks=liger_path_trace \
  model.root.candidate_trace=true \
  model.root.path_trace=true
```

统一入口在完整 `data/beauty/testing` 上通过Trainer.predict输出推荐，沿用beam20加全部cold与content终排，保留checkpoint中的学习alpha。输出包含user_id键，并记录test/recall@5、test/recall@10、test/ndcg@5、test/ndcg@10和test/user_count。发布recommendation_output、liger_candidate_trace及liger_path_trace，后两项用于定位新候选、丢失候选与具体剪枝层；开启trace不改变候选搜索或排序。

命令已在node1通过Bash语法检查、uv函数替身参数展开和Hydra compose；确认精确checkpoint URI、full testing数据、v3模型、beam20、学习alpha及两个trace writer。Mutagen flush成功，三个session均Watching for changes，无conflict。

## 效果评价边界

此次没有启动模型前向、完整推理或续训。训练结果已登记，testing收益和候选净收益仍未确认；不能用validation数值替代相对真实v0/LIGER dense的testing结论。用户运行后，以新inference run核验输入/checkpoint lineage、完整用户/标签及输出合法性，独立复算Recall/NDCG，再比较匹配Beauty/seed42的真实v0与LIGER dense，并通过trace分解候选恢复/损失和共同命中位置变化。既有testing开发使用和关闭预算边界继续保留。

核验数据见 [W&B快照](evidence/copmrec-v3-rbha00vx-20261004/wandb-run.json)、[checkpoint核验](evidence/copmrec-v3-rbha00vx-20261004/checkpoint-audit.json) 和 [命令核验](evidence/copmrec-v3-rbha00vx-20261004/single-gpu-command-audit.json)。
