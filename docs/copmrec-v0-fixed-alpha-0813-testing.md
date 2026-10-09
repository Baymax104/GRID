# v0 固定推理 alpha=0.813：单卡对照

> 2026-10-05：用户已取消此尚未运行的完整推理，剩余额度0。下方命令仅保留准备历史；当前任务为 [v0固定alpha=0.8从头训练](copmrec-v0-fixed-alpha08-training.md)。

日期：2026-10-04。用户要求将v0固定alpha为0.813看效果。复用真实v0的7y54j4m6 best48000，只使用已有推理覆盖字段，不修改代码、训练权重或默认值。一次用户手动完整Testing，零训练、零扫描；保留v3.1作为当前开发基座。

## 对照与判断边界

对照为真实v0推理6dspa7e3，学习alpha=0.8132556080818176，Recall10=0.071904485、NDCG10=0.036448314。精确固定值0.813与学习值只差−0.000255608；本次检验该细小推理权重变化的效果，不能回答“固定权重训练是否优于学习权重训练”。候选和预测可能基本相同，不能预先宣称提升。

保持全层Mass、合法条件概率混合、同一SID/embedding、seed42、FP32、beam20加cold33、content Top10、full Testing和predict batch32。开启已有candidate/path trace用于损益审计，标签不参与候选选择；不引入v3/v3.1 Max或root包络，不合并其他模型候选。

checkpoint producer7y54j4m6，best step48000；引用 `liger_beauty_joint_train-checkpoint-local-reference:v2`，digest `ebed11b2799e2829b30df32fed991166`，文件MD5/base64 `P6FET6ef+gLprgX0lwStRw==`。该引用为历史字节一致迁移，保留已有provenance边界，不以本次推理补造早期训练源码。

## 单卡命令

在node1仓库根目录 `/data3/weizhenyu/projects/GRID` 执行。物理GPU2→logical GPU0，单进程。根脚本通过统一 `uv run -m src.main`，默认source snapshot归档本次实际运行字节。

```bash
export PATH="$HOME/.local/bin:$PATH"

COPMREC_V0_BEST_CKPT='wandb://baymaxam/GRID/7y54j4m6?role=checkpoint&alias=v2&file=checkpoint_epoch=000_step=048000.ckpt'

CUDA_VISIBLE_DEVICES=2 NPROC_PER_NODE=1 bash copmrec_v0_inference.sh \
  --dataset beauty \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --checkpoint "$COPMREC_V0_BEST_CKPT" \
  --devices '[0]' \
  --seed 42 \
  --master-port 29549 \
  --group copmrec_v0_alpha0813_7y54j4m6_testing \
  --notes 'CoPMRec v0 fixed-alpha对照：复用7y54j4m6 best48000，仅推理alpha精确固定0.813；原学习值0.813255608，全部SID层Mass、单beam20加全部cold、content终排；对照6dspa7e3最终指标与候选交换；单次推理、零训练；Testing已用于开发分析' \
  model.root.inference_mixture_alpha=0.813 \
  model.root.path_trace=true \
  callbacks=liger_path_trace
```

## 验证与结果门禁

准备核验已完成：9项相关测试通过；完整命令的shell语法、参数透传及Hydra compose通过。Mutagen flush成功，三个session均Watching且无conflict；远端12个相关运行文件SHA256与本地一致。W&B实时核对上述v2引用、producer及文件digest；真实best48000严格加载164个state tensor，固定覆盖前后全部相同。与6dspa7e3记录的模型配置仅推理alpha有差异，数据配置完全相同，seed42、FP32与source snapshot启用已核对。没有进行真实模型前向、候选搜索、训练或完整推理。

核对实际checkpoint与source、相同输入/全体用户/dense排名、trace的 `content_mixture_alpha=0.813` 和 `mixture_alpha_source=fixed_inference`。独立复算Recall/NDCG@5/10，对照6dspa7e3统计候选集合/路径变化、恢复与损失、共同命中位置，并作固定checkpoint用户配对区间。主要评价NDCG10，同时看Recall10与净命中。

正向净改善支持该checkpoint的固定推理设置，不推广为训练方式或最优alpha结论；负向保持原学习值；输出相同或区间跨零记录无可确认收益，不自动扫参或追加预算。一次推理由用户手动启动，完整效果待验证。准备证据见 [verification.json](evidence/copmrec-v0-alpha0813-20261004/verification.json)。
