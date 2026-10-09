## 6. 执行步骤与命令

入口已实现并完成本地验证，正式运行未启动。从 GRID 根目录运行；物理 GPU 0 → CUDA local [0]，NPROC=1。M1 不做模型 forward，单卡设置与其他诊断保持一致。

五个变体 Testing 完成后填写各自不可变 bundle URI；Full 的精确输出已填。加载通过共享 Artifact reader，六份输出按业务 user keys 配对，全集不一致、非法/重复/历史重叠输出会拒绝；不从行号推断配对。五个待填引用尚未存在，不使用 Full 输出冒充。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export COPMREC_FULL_OUTPUT='wandb://baymaxam/GRID/vosmuihm?role=recommendation_output&alias=v0&file=merged_predictions_tensor.pt'
export COPMREC_NO_MIXTURE_OUTPUT='【待填写：BMX-122 Testing 的不可变 recommendation_output URI】'
export COPMREC_NO_RESIDUAL_OUTPUT='【待填写：BMX-123 Testing 的不可变 recommendation_output URI】'
export COPMREC_NO_NATIVE_OUTPUT='【待填写：BMX-129 Testing 的不可变 recommendation_output URI】'
export COPMREC_LEGAL_GENERATION_OUTPUT='【待填写：BMX-142 Testing 的不可变 recommendation_output URI】'
export COPMREC_JOINT_CE_REPLACE_OUTPUT='【待填写：BMX-143 Testing 的不可变 recommendation_output URI】'

bash ./copmrec_diagnosis.sh \
  --analysis hits --variant full \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --dataset beauty --devices '[0]' --seed 42 \
  --group paper_mechanism_copmrec_beauty \
  --notes 'issue=BMX-144; protocol=copmrec-m3-v53-20261008-v1; analysis=hits; dataset=beauty; seed=42; paired-testing-bundles; no-model-forward' \
  +prediction_paths.full="\"$COPMREC_FULL_OUTPUT\"" \
  +prediction_paths.no_mixture="\"$COPMREC_NO_MIXTURE_OUTPUT\"" \
  +prediction_paths.no_residual="\"$COPMREC_NO_RESIDUAL_OUTPUT\"" \
  +prediction_paths.no_native="\"$COPMREC_NO_NATIVE_OUTPUT\"" \
  +prediction_paths.legal_generation="\"$COPMREC_LEGAL_GENERATION_OUTPUT\"" \
  +prediction_paths.joint_ce_replace="\"$COPMREC_JOINT_CE_REPLACE_OUTPUT\""
```

JSON/CSV/keys-predictions 及完整 manifest 由 epoch 共享 writer 一次写入 paths.output_dir/analysis，并发布最终分析 Artifact；没有上传中间缓存。差值为 variant-Full，保留全部预定切片及空组 null、固定哈希选例，独立核对 NDCG 可加分解。
