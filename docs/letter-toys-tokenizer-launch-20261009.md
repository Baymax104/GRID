# LETTER Toys Tokenizer 启动记录（2026-10-09）

## 启动核验快照

核验时间（UTC）：`2026-10-09T14:28:59.463123+00:00`。三个run均running，无exit文件，实际step>0且loss有限；resolved config、消费上游artifacts、源码origin和源码archive SHA256检查通过。

| seed | epoch | trainer/global_step | train/loss |
|---:|---:|---:|---:|
| 42 | 104 | 1249 | 0.096884 |
| 200 | 79 | 949 | 0.091434 |
| 2026 | 95 | 1149 | 0.105203 |

BMX-66～68已更新为Tokenizer训练中，保持In Progress。快照不代表当前实时进度；首个正式collision_rate验证在第2000个epoch结束时执行。

用户授权在 node1 的独立 tmux 会话中启动三个 seed 的正式 Tokenizer 训练。各自消费对应 seed 已核验的 CF 导出；内容目录固定为11924商品、1024维，CF为同目录32维。

## 运行分配

| Issue | seed | 物理 GPU → CUDA | 端口 | W&B run | tmux |
|---|---:|---|---:|---|---|
| BMX-66 | 42 | 2 → 0 | 30266 | [c6pirdxb](https://wandb.ai/baymaxam/GRID/runs/c6pirdxb) | `letter_tokenizer_bmx66_toys_s42_c6pirdxb` |
| BMX-67 | 200 | 7 → 0 | 30267 | [kiqf0416](https://wandb.ai/baymaxam/GRID/runs/kiqf0416) | `letter_tokenizer_bmx67_toys_s200_kiqf0416` |
| BMX-68 | 2026 | 1 → 0 | 30268 | [dcv99aeo](https://wandb.ai/baymaxam/GRID/runs/dcv99aeo) | `letter_tokenizer_bmx68_toys_s2026_dcv99aeo` |

## 固定协议与来源

单卡 FP32，batch1024；最多20000 epoch，每2000 epoch按完整目录 val/collision_rate 最小选best，last仅用于恢复。latent32、4×256、10groups，alpha0.01/beta0.0001/mu0.25；AdamW lr0.001、WD0.0001。正式命令显式input_dim=1024，ckpt_path=null，run_test_after_training=false。

交付前Mutagen flush成功，三个session均Watching for changes、无conflict。使用已修复diversity采样的独立LETTER Tokenizer；tokenizer.py SHA256=`aa8a4d0aaae9b086b53c3716dead27e20293d6249b28f4d64c94c8a18f59bf77`。运行源码SHA256=`45635c00cba54eb43df3582c163ebde52dce53130198d98f75a6df2cb552c2b9`，source origin verified；三个source artifact均COMMITTED。

- BMX-66：CF=`wandb://baymaxam/GRID/jx7hith4?role=collaborative_embedding&alias=v2&file=merged_predictions_tensor.pt`；来源best=`wandb://baymaxam/GRID/wsflubzp?role=checkpoint&alias=v2&file=step_step=039000.ckpt`。
- BMX-67：CF=`wandb://baymaxam/GRID/ljowhf8y?role=collaborative_embedding&alias=v1&file=merged_predictions_tensor.pt`；来源best=`wandb://baymaxam/GRID/yix316dz?role=checkpoint&alias=v0&file=step_step=044000.ckpt`。
- BMX-68：CF=`wandb://baymaxam/GRID/dwkqg7ud?role=collaborative_embedding&alias=v0&file=merged_predictions_tensor.pt`；来源best=`wandb://baymaxam/GRID/q7cblret?role=checkpoint&alias=v1&file=step_step=043000.ckpt`。

## 复现与查看

node1仓库根目录：`/data3/weizhenyu/projects/GRID`。精确参数、notes、输入URI、GPU/端口与输出目录保存在 `logs/letter-toys-tokenizer-20261009/tokenizer-plan.json`；脚本已通过bash语法检查。各会话包含train/logs窗口，退出时记录独立exit文件。

```bash
tmux attach -t letter_tokenizer_bmx66_toys_s42_c6pirdxb
tmux attach -t letter_tokenizer_bmx67_toys_s200_kiqf0416
tmux attach -t letter_tokenizer_bmx68_toys_s2026_dcv99aeo 
```

对应启动脚本为 `logs/letter-toys-tokenizer-20261009/tokenizer-bmx66.sh`、`tokenizer-bmx67.sh`、`tokenizer-bmx68.sh`；执行前先检查现有run，避免重复启动。启动核验文件为同目录 `startup-verification.json`。本记录仅证明启动，不代表训练完成、SID合法或最终推荐效果成立。
