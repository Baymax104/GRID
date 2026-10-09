# LETTER SID 导出与推荐训练启动记录

> 2026-10-09 更新：六组推荐训练已停止，所对应的 W&B run 与六个源码 Artifact 已按用户要求删除。本文运行状态与链接为历史记录；清理核验见 [停止与清理记录](letter-cancel-20261009.md)。CF、Tokenizer、SID 上游保留。

日期：2026-10-08（Asia/Shanghai）。用户授权在 node1 完成 Beauty/Sports 的六组 SID 导出并在 tmux 启动推荐训练；不包含最终 Testing。

## SID 导出

采用 BMX-60～65 中已固定的 validation-selected Tokenizer best checkpoint，保留原 content/CF/source；统一入口为根目录 letter_sid.sh → src.main，物理 GPU1、单进程、FP32。

六组均正常结束。Beauty 的三组 bundle 为 12101×4，Sports 为 18357×4，商品 key 与各自 content/CF 完整目录逐项一致，四码均为整数且范围 [0,256)，全局唯一。独立 CUDA/medium 原始编码比对确认前三码不变；完整 prefix 人口均不超过256。

| Issue | SID run | Artifact | 最大 prefix 人口 | 末码变化数 |
| --- | --- | --- | --- | --- |
| BMX-60 | [3fgw74y3](https://wandb.ai/baymaxam/GRID/runs/3fgw74y3) | letter_sid_beauty-semantic-id:v0 | 35 | 60 |
| BMX-61 | [ilfqdvq3](https://wandb.ai/baymaxam/GRID/runs/ilfqdvq3) | letter_sid_beauty-semantic-id:v1 | 20 | 28 |
| BMX-62 | [842jwvqs](https://wandb.ai/baymaxam/GRID/runs/842jwvqs) | letter_sid_beauty-semantic-id:v2 | 35 | 62 |
| BMX-63 | [n19hvxe1](https://wandb.ai/baymaxam/GRID/runs/n19hvxe1) | letter_sid_sports-semantic-id:v0 | 65 | 83 |
| BMX-64 | [7gbg8agr](https://wandb.ai/baymaxam/GRID/runs/7gbg8agr) | letter_sid_sports-semantic-id:v1 | 14 | 195 |
| BMX-65 | [1ec5jvzl](https://wandb.ai/baymaxam/GRID/runs/1ec5jvzl) | letter_sid_sports-semantic-id:v2 | 14 | 124 |

实际 checkpoint selection=best / monitor=val/collision_rate / mode=min 与已选版本一致；SID run 消费 checkpoint、CF 和 content 的 Artifact lineage 逐组通过。公共 loader 与 LetterCatalog 输入校验通过；本地 writer 与 W&B writer 的 bundle 逐项一致。源码 archive 和 manifest 逐文件 size/SHA256 核验通过，origin=verified，六组 source SHA256 均为 5e95cf7e87b28b730976b5a6d279acdc7aca3a270a08dd5e4519ce1fc6624867。

## 推荐训练

通过根目录 letter_train.sh → uv run torchrun -m src.main，六组均为双卡 DDP、每卡 batch128/global256、FP32、history20、T5 随机初始化、temperature1/beam20/top10。50k optimizer step，每500步全量 evaluation，以 val/ndcg@10 最大选 best；不自动 Testing。没有更改冻结参数、虚拟环境或其他用户的 GPU 任务。三组 GPU pair 各承载 Beauty/Sports 两个实验。

| Issue | Dataset / Seed | GPU | W&B run | 启动核验时已记录 global_step | 训练 loss | tmux session |
| --- | --- | --- | --- | --- | --- | --- |
| BMX-60 | beauty / 42 | 0,1 | [vzzddvhg](https://wandb.ai/baymaxam/GRID/runs/vzzddvhg) | 499 | 6.384355 | letter_bmx60_beauty_s42_20261008 |
| BMX-61 | beauty / 200 | 2,3 | [oremp4gb](https://wandb.ai/baymaxam/GRID/runs/oremp4gb) | 249 | 7.353255 | letter_bmx61_beauty_s200_20261008 |
| BMX-62 | beauty / 2026 | 4,5 | [od6uvwdi](https://wandb.ai/baymaxam/GRID/runs/od6uvwdi) | 299 | 7.002093 | letter_bmx62_beauty_s2026_20261008 |
| BMX-63 | sports / 42 | 0,1 | [h628mv5l](https://wandb.ai/baymaxam/GRID/runs/h628mv5l) | 499 | 5.821748 | letter_bmx63_sports_s42_20261008 |
| BMX-64 | sports / 200 | 2,3 | [gzi64g6z](https://wandb.ai/baymaxam/GRID/runs/gzi64g6z) | 449 | 6.319804 | letter_bmx64_sports_s200_20261008 |
| BMX-65 | sports / 2026 | 4,5 | [xzf4m2f0](https://wandb.ai/baymaxam/GRID/runs/xzf4m2f0) | 399 | 6.812490 | letter_bmx65_sports_s2026_20261008 |

六组均核验 W&B state=running、global_step>0、loss 有限，配置符合上述冻结条件，各自 SID 的实际消费 lineage 正确；日志没有 traceback/OOM。训练源码 archive/manifest 核验通过，source 与 SID 导出一致且 origin=verified。表中 step 是启动检查的历史快照，不代表当前实时进度或最终有效性。未等待50k训练结束，未产生或验证最终 Testing 指标。

## 查看与记录

在 node1 使用：

```bash
tmux attach -t letter_bmx60_beauty_s42_20261008
# Ctrl-b d 脱离会话，训练继续。
```

每个 session 保留训练窗口和独立 logs 窗口，logs 窗口默认 tail -F 对应日志。

node1 记录目录：/data3/weizhenyu/projects/GRID/logs/letter-formal-20261008-8975393/。每组 sid-bmxNN.sh / train-bmxNN.sh 保存实际执行命令，*.log 保存运行输出，正常或失败退出后写入相应 *.exit。六组 SID exit 均0，启动核验时六组训练均未退出。

本地证据副本：logs/letter-formal-20261008-8975393/{sid-verification,training-plan,training-startup-verification}.json。SID 固定 URI、catalog/bundle/source SHA256 与消费 Artifact 均在 JSON 中登记。

