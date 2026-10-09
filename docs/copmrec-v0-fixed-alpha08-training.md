# v0固定alpha=0.8从头训练

日期：2026-10-05。用户取消尚未运行的v0固定推理0.813对照，改为一次v0固定alpha=0.8从头训练。当前v3.1开发基座和过去结果保留。

训练已由用户完成，run ncyol9n0通过配置、固定alpha、实际源码和best checkpoint核验，50k完成日志明确；best44500的dense验证NDCG10=0.046339214。单次训练额度已消费，后续单卡Testing命令见 [训练核验与推理说明](copmrec-v0-fixed-alpha08-ncyol9n0-testing.md)。下方保留准备时的配置及验证边界。

## 实现与对照

`fixed_mixture_alpha=0.8`从随机初始化开始设置全局gate的logit并冻结，所有样本与SID层共用；三项损失仍为SID CE+content CE+合法条件混合NLL，两路模型保持可训练。训练、验证和推理共用冻结gate，全层Mass，候选beam20加全部cold、content终排。普通v0默认仍学习alpha；已有`inference_mixture_alpha`仅推理字段与固定训练策略互斥。

固定策略checkpoint记录`copmrec_alpha_policy=fixed`及`copmrec_fixed_mixture_alpha=0.8`，拒绝加载学习策略或其他固定值，严格加载检查冻结bias。推理必须装配相同固定训练模型，例如通过已有v0推理入口覆盖`model.root.fixed_mixture_alpha=0.8`，不能用推理专用字段替代；训练run审核并选定best后再交付独立Testing命令。

研究定位是v0全局权重的训练消融，缩小“固定权重下联合训练能否改善最终推荐”的不确定性，不能推出0.8最优或固定训练普遍优越。对照真实v0训练7y54j4m6及推理6dspa7e3，原alpha约0.813255608。固定训练与仅改推理具有不同训练轨迹，过去固定推理结果不能替代此实验。

## 训练配置

继承v0全部训练配置：Beauty、seed42、SID `dq77e3wo`、embedding `3jtt9mpa`；T5和内容投影随机初始化，`ckpt_path=null`。训练每卡batch128、双卡总batch256、累积1，50k optimizer更新；AdamW lr=0.0003、weight_decay=0.035，warmup2500加cosine至50k，FP32，clip1.0；每500批验证，全量dense `val/ndcg@10`选best。训练后不自动Testing，正式源码快照默认开启。

## 双卡命令

从node1仓库根目录`/data3/weizhenyu/projects/GRID`执行。物理GPU2/3映射为进程内GPU0/1；不加载旧checkpoint，不默认dry-run。

```bash
export PATH="$HOME/.local/bin:$PATH"

CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 bash copmrec_v0_fixed_alpha_train.sh \
  --dataset beauty \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --devices '[0,1]' \
  --seed 42 \
  --master-port 29551 \
  --group copmrec_v0_fixed_alpha08_scratch \
  --notes 'CoPMRec v0固定alpha训练消融：随机初始化，alpha从训练开始固定0.8并冻结，全部SID层Mass，三项损失等权；其余配置与v0一致，GPU2/3、每卡batch128、累积1、50k更新、dense验证选best；对照7y54j4m6/6dspa7e3；单次训练，不扫描alpha；Testing已用于开发分析' \
  ckpt_path=null
```

## 验证与预算

共121项相关单测通过（新增与v0/v3/v3.1检查79项，v1/v1.1/v2恢复回归42项），覆盖独立概率枚举、teacher forcing/增量解码、两路梯度、固定gate优化、RNG初始化一致、恢复/拒绝、trace、Hydra配置及脚本quoting/空值/非法参数/末尾override/torchrun。实时核对W&B的7y54j4m6配置：数据和optimizer/scheduler完全一致，trainer仅输出目录变化，模型仅固定alpha及显式关闭历史缺省path_trace。Mutagen flush成功、三会话Watching且无conflict，远端16个相关文件SHA一致；完整正式命令mock/compose和shell语法通过。

双卡显式零学习率单步dry-run成功，物理2/3→logical0/1、NCCL两进程，日志train/mixture_alpha=0.8000。此装配验证每卡batch2、warmup0、max_steps1，W&B/保存checkpoint关闭，不作效果或正式batch128显存证据；正式命令使用原v0 batch128/warmup2500/50k更新。完整训练未启动。证据见 [verification.json](evidence/copmrec-v0-fixed-alpha08-20261005/verification.json)。

授权一次用户手动训练，零扫描，不重置旧关闭预算；原未启动的0.813推理取消，剩余0。主要效果指标为同协议Testing NDCG10，联合Recall10、候选新增/损失及配对区间；按原dense验证选点，不按Testing挑checkpoint。正向支持此设置的有限证据，负向/不确定保留学习默认，不自动新增运行。训练完成后审计实际源码、配置、曲线、alpha恒定及最佳checkpoint，再准备单卡Testing。
