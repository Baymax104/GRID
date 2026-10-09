# v0固定alpha0.8训练核验与单卡Testing

日期：2026-10-05。训练run [ncyol9n0](https://wandb.ai/baymaxam/GRID/runs/ncyol9n0) 完成；本次只读审计及命令准备，不启动训练/完整推理，不写外部W&B或Linear。

Testing已由用户完成，run x70hphts独立核验通过；Recall10/NDCG10相对学习alpha v0下降3.05%/2.60%，本设置不晋级。单次训练/Testing已消费，不追加预算。结果见 [完整Testing与bad case报告](copmrec-v0-fixed-alpha08-x70hphts-result.md)，下方保留当时命令准备记录。

## 训练核验

- 正式配置与已准备固定0.8训练配置的data/model完全相同，seed42、ckpt_path=null，双卡每卡128、累积1、50k更新、FP32、三项等权损失、全层Mass；optimizer/scheduler与原v0相同。
- W&B训练alpha1000条均为0.800000011920929；验证alpha100条均为0.8000009059906006，为指标聚合的FP32舍入，checkpoint gate符合冻结初始化值。日志明确`max_steps=50000`完成，W&B global_step摘要49999属于记录时点。
- 100次dense验证中的最大NDCG10=0.046339213848114014；best44500的callback状态、Artifact metadata及完整history最大值一致。比原v0 best48000的0.04743017628788948低约2.30%；该比较是dense验证指标，最终hybrid Testing效果待运行。
- best checkpoint v0协议、fixed策略、固定值0.8、文件MD5/大小及全部状态有限已验证。`last.ckpt`实际保存step44500，不能用来证明末步或作为50k恢复点；本次命令固定验证best44500，50k执行完成由终止日志证明。
- 输入SID/embedding的producer/digest与原v0一致：dq77e3wo/19ace08283163fbe687287fbacaa3842、3jtt9mpa/ab56af975eac589c27eed6094482cbd7。
- 实际source snapshot的342个文件逐项SHA/大小及tar/manifest/aggregate hash全部通过，与准备时16个相关运行文件一致；source_sha256=dc71443d4740403c1bfa0493974f8288e3d08ed55bb74bb5fbb57015b7f25b7b。

checkpoint Artifact为`copmrec_beauty_v0_fixed_alpha08_train-checkpoint:v0`，producer ncyol9n0，digest `5e490d391fafee0d086d3275eacd9f94`；文件`checkpoint_epoch=000_step=044500.ckpt`，MD5/base64 `aEa/LzeefPLMh33BZFtTEg==`，164258219字节。

## 单卡命令

从node1仓库根目录`/data3/weizhenyu/projects/GRID`执行。物理GPU2→logical GPU0，predict batch32、seed42、FP32、beam20加cold33、content Top10、全量Testing。使用`fixed_mixture_alpha`恢复训练策略，推理专用覆盖保持null。

```bash
export PATH="$HOME/.local/bin:$PATH"

COPMREC_V0_FIXED08_BEST_CKPT='wandb://baymaxam/GRID/ncyol9n0?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=044500.ckpt'

CUDA_VISIBLE_DEVICES=2 NPROC_PER_NODE=1 bash copmrec_v0_inference.sh \
  --dataset beauty \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --checkpoint "$COPMREC_V0_FIXED08_BEST_CKPT" \
  --devices '[0]' \
  --seed 42 \
  --master-port 29555 \
  --group copmrec_v0_fixed_alpha08_ncyol9n0_testing \
  --notes 'CoPMRec v0固定alpha0.8从头训练Testing：ncyol9n0验证选best44500，恢复fixed训练策略0.8，全层Mass、beam20加全部cold、content终排；同SID/embedding、seed42、FP32、全量Testing；对照7y54j4m6/6dspa7e3学习alpha训练；单卡推理，Testing已用于开发分析' \
  model.root.fixed_mixture_alpha=0.8 \
  model.root.path_trace=true \
  callbacks=liger_path_trace \
  task_name=copmrec_beauty_v0_fixed_alpha08_inference
```

## 准备核验与效果判断

推理命令shell语法、mock参数透传、Hydra配置、URI解析及真实checkpoint严格加载均通过，164个state tensor逐项一致，gate保持冻结。训练到推理root配置仅改变training_model_config（设null）、candidate/path trace观测开关。Mutagen flush完成、三会话Watching且无conflict，远端12个推理相关文件与本地SHA一致；未进行真实模型前向、候选搜索或完整推理。开启已有candidate/path trace作为损益分析观测，source snapshot默认开启。

完成后核验actual source、config、fixed_training trace、输入digest、完整22363用户身份和合法输出，独立复算Recall/NDCG@5/10，并与6dspa7e3统计候选恢复/损失/共同命中位置及配对区间。正向支持此固定训练设置的有限证据；负向或不确定保留学习默认，不自动扫描或追加训练。验证选点保持dense规则，不按Testing选checkpoint。

证据见 [verification.json](evidence/copmrec-v0-fixed-alpha08-ncyol9n0-20261005/verification.json)。原0.813推理继续取消，旧预算不重置；一次固定训练额度已消费，当前v3.1开发基座保留。
