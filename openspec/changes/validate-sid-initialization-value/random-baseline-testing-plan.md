# 随机初始化seed42冻结testing补充

## 范围

仅新增一项mask_ce seed42 testing，复用已有训练y1dnupbt的best-val 19000-step checkpoint。A的testing直接复用sb6eapjd，不训练任何模型，不重跑A，不使用五seed均值对单seed基准。

checkpoint文件checkpoint_epoch=000_step=019000.ckpt；Artifact digest14bee23f40882bef98b19685759afb35；SID digest20f08b323a286fbb3f16b5ea27562af1；内容digestab56af975eac589c27eed6094482cbd7。A checkpoint digest为aaba5b47dc9a5e94c0d23b75cf697238；A testing recommendation digest为f312a292cd4cd6316eaba3bf765cf47f，prefix trace digest为e59c145db70ae2aab2985ac1c78d09d3。

复用原tiger_catalog_grounded_inference.sh，arm=mask_ce；不要使用五seed A/残差包装脚本，该脚本不包含随机基准。预期checkpoint digest为结果审计字段，不替代实际lineage核验。

## 手动命令

```bash
cd /data3/weizhenyu/projects/GRID
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash ./tiger_catalog_grounded_inference.sh \
  --data-dir data/beauty --dataset beauty --data-split testing \
  --semantic-id-path 'wandb://baymaxam/GRID/4vyi4o6w?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --checkpoint-path 'wandb://baymaxam/GRID/y1dnupbt?role=checkpoint&file=checkpoint_epoch=000_step=019000.ckpt' \
  --arm mask_ce --group rkmeans --devices '[0]' \
  --beam-width 10 --seed 42 --master-port 29772 \
  --notes 'Frozen matched random initialization; seed42; exploratory testing supplement; reuse A sb6eapjd; no training or tuning' \
  model=tiger_catalog_grounded_inference \
  task_name=sid_frozen_mask_ce_beauty_seed42_testing \
  +testing_protocol=beauty-random-init-seed42-v1 \
  +initialization_condition=random_mask_ce \
  +testing_training_seed=42 \
  +testing_training_run=y1dnupbt \
  +testing_expected_checkpoint_digest=14bee23f40882bef98b19685759afb35 \
  +testing_reference_inference_run=sb6eapjd \
  +testing_physical_gpu=0
```

## 判读

先核对finished、lineage、testing、相同data配置、用户key/标签及合法唯一beam10预测。只与A seed42比较NDCG@10、Recall@10、命中增失及同目标前两层SID cluster bootstrap（2000次、seed42）。A已有testing NDCG0.03190617697480039、Recall0.06179850646156598，22363用户；最终应从两份预测重新成对计算，不混用evaluation。

该补充是在A/残差testing结果已知后提出，属于探索性对照，不称预先注册确认。历史testing基线仅用于报告的用户确认仍保留。正向结果仅支持单seed，不能代表跨seed稳定收益；只有收益保持且没有明显Recall代价才考虑补齐四个随机基准训练，若负向或不明确则停止扩展。不得用本轮testing调整模型/选择checkpoint。

本轮生产实现与配置无需修改，使用既有入口和历史兼容的mask_ce checkpoint契约。完整实验由用户手动启动。
