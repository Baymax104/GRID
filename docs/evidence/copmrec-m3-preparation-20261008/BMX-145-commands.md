## 6. 执行步骤与命令

入口已实现并完成本地验证，正式运行未启动。从 GRID 根目录运行；物理 GPU 0 → CUDA local [0]，NPROC=1。Full own-best 精确 URI/SHA 和既有 Testing bundle 已填。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export COPMREC_BEST_CKPT='wandb://baymaxam/GRID/gshpyn49?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=047500.ckpt'
export COPMREC_CHECKPOINT_SHA256='a64ca93d49b5bb9ead7b4af803091346ca62af5299cd89266ac5a2acce3894c4'
export COPMREC_FULL_OUTPUT='wandb://baymaxam/GRID/vosmuihm?role=recommendation_output&alias=v0&file=merged_predictions_tensor.pt'

bash ./copmrec_diagnosis.sh \
  --analysis residual --variant full \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --dataset beauty --devices '[0]' --seed 42 \
  --group paper_mechanism_copmrec_beauty \
  --checkpoint "$COPMREC_BEST_CKPT" \
  --checkpoint-sha256 "$COPMREC_CHECKPOINT_SHA256" \
  --notes 'issue=BMX-145; protocol=copmrec-m3-v53-20261008-v1; analysis=residual; source_variant=full; dataset=beauty; seed=42; own-validation-best; eval-only; empirical-observations' \
  +prediction_paths.full="\"$COPMREC_FULL_OUTPUT\""
```

先核对 SHA、Full 恢复契约及 strict state，再在 eval/无更新下测量四视图。V11 重新计算只用于与既有 Full bundle 精确核对，不作为额外独立实验；V10/V01/V00 为三个新增评分视图。h=0 必须重新 encode，固定 query 时检查 cold 原始 logits 的 c 切换不变。保留 eligible=false 用户，其 rank/margin 为 null；报告四指标、相对 margin、Top10 overlap、norm、差值/CI及交互。共享 epoch writer 发布最终 JSON/CSV/bundle/manifest，不上传中间缓存。
