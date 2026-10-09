## 6. 执行步骤与命令

入口已实现并完成本地 CPU/配置/脚本验证；本次没有启动正式实验。全部命令从 GRID 仓库根目录执行，沿用现有实验 issue 的环境变量、参数、notes 和 quoted Artifact 写法。物理 GPU 0,1 → CUDA local [0,1] 用于训练；物理 GPU 0 → local [0] 用于 Testing。GPU/端口是可调整的资源示例，不表示已分配。

### 训练

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=29523

bash ./copmrec_ablation_train.sh \
  --variant no_native \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --dataset beauty --devices '[0,1]' --seed 42 \
  --group paper_ablation_copmrec_beauty \
  --notes 'issue=BMX-129; protocol=copmrec-m3-v53-20261008-v1; variant=no_native; dataset=beauty; seed=42; split=train; scratch; empirical-observations'
```

### Testing

训练结束后先审计本臂首次最高 raw val/ndcg@10 的 own-best，登记不可变版本/文件/SHA；替换下方两个待填值再独立执行。禁止使用 Full、其他变体或 last.ckpt 替代。脚本会拒绝待填 SHA，不会自动选 checkpoint。

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export COPMREC_BEST_CKPT='【待填写：BMX-129 自身训练的 validation-selected best 不可变 URI】'
export COPMREC_CHECKPOINT_SHA256='【待填写：该 best 文件的 SHA256】'

bash ./copmrec_ablation_inference.sh \
  --variant no_native \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --dataset beauty --devices '[0]' --seed 42 \
  --group paper_ablation_copmrec_beauty \
  --checkpoint "$COPMREC_BEST_CKPT" \
  --checkpoint-sha256 "$COPMREC_CHECKPOINT_SHA256" \
  --split testing \
  --notes 'issue=BMX-129; protocol=copmrec-m3-v53-20261008-v1; variant=no_native; dataset=beauty; seed=42; split=testing; own-validation-best; empirical-observations'
```

新输出按 keys/predictions bundle 发布至该 Testing run。训练与 Testing 的 group 相同；variant/seed/stage 另记 W&B config、tags、notes。支持 --dry-run、--notes 两种写法及额外 Hydra override；本次只使用 uv 参数 stub 和 compose 验证，没有执行真实 dry-run。运行前按 Mutagen 契约同步并登记数据 manifest/Artifact digest；runtime 原始源码快照由统一入口归档。
