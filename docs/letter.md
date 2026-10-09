# LETTER 独立实现与运行协议

## 实现边界

对照[作者 LETTER 仓库](https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda)，固定 commit `8d0154e28de37dbb6e24871c508ad8ddb1921cda`。接入对象是 LETTER-TIGER，不包含 LETTER-LC-Rec。

模型完全独立：`src/quantization/letter/` 实现内容重建、残差量化、CF 对比、diversity、constrained K-means 与 SID 修复；`src/recommendation/letter/` 实现随机初始化标准 T5、温度训练、Trie 约束生成和独立 item 评价。没有调用项目 RQ-VAE、TIGER、SASRec、LIGER 或 CoPMRec 模型。复用范围为 Lightning、Hydra、公共配置容器、TFRecord reader、FileDataModule、MetricEngine、scheduler、artifact resolver 与共享 writer。

分别提案并实现七个 OpenSpec change：`add-letter-tokenizer`、`add-letter-recommender`、`add-letter-data`、`add-letter-training-evaluation`、`add-letter-pipeline`、`add-letter-cf-teacher`、`fix-letter-sid-collision-export`。

## 对齐与明确差异

| 项目 | 当前实现 |
| --- | --- |
| tokenizer | 四级学习码，每级256，latent32；MLP 2048/1024/512/256/128/64 |
| loss | 重建 MSE + 四层平均 VQ + alpha × batch CF dot-product CE；每层 VQ 包含 beta × diversity |
| 梯度 | 保留作者逐层 STE；CF 不直接更新码本，后层 commitment 不回传 encoder |
| 初始化/分组 | 完整目录 constrained K-means，n_init10/max_iter10/n_jobs10；每 epoch 分组，排除 self positive |
| tokenizer 训练 | AdamW lr0.001、WD0.0001、batch1024，最多20000 epoch，每2000 epoch按完整目录碰撞率选优 |
| SID 导出 | 完整目录最近邻，末层 Sinkhorn 最多20轮；容量内剩余碰撞以全prefix末层最小距离硬匹配，超容量拒绝，不追加第五位 |
| 推荐模型 | T5 128/1024、4层、6头、d_kv64、dropout0.1、绑词嵌入，随机初始化 |
| token 编号 | 原始词表32100之后按观察到的 `<a_i>` 等字符串排序添加；EOS1、start/pad0 |
| 训练监督 | 每个非空历史前缀监督下一商品；四码目标加EOS；训练logits除temperature，默认1.0 |
| 生成 | 原始logits、beam20、完整目录Trie、length_penalty1，不重新归一化，不过滤历史 |
| 共同协议 | training/evaluation/testing，history20，FP32，2GPU每卡128，accumulation1，50k step，每500 step验证 |
| 推荐优化器 | 保留作者 AdamW lr0.0005、WD0.01；bias/LayerNorm不decay，1% warmup（500步）和cosine到0，gradient clip1 |
| 选优与评价 | 按 val/ndcg@10 最大选checkpoint；全目录Recall/NDCG@5/@10，所有用户及cold target保留 |

共同数据与评价约束来源为 [BMX-58](https://linear.app/baymax104/issue/BMX-58/准备letter-适配冻结协议与可执行命令) 与 [BMX-6](https://linear.app/baymax104/issue/BMX-6/冻结数据训练与评价协议)。BMX-6 的核心匹配训练只针对 CoPMRec/LIGER；LETTER方法协议按用户本次完成BMX-58要求冻结为两卡每卡128、50k step/500 step验证，方法专用lr、WD取作者默认；没有新增调参搜索预算。将作者推荐端的 epoch/validation-loss 选优改为 step/NDCG 协议，增加独立CF teacher预算及容量内SID硬匹配；这些为明确的GRID适配，不宣称逐项复刻论文训练预算或输出。SID导出协议为 `letter-sinkhorn-prefix-assignment-v1`，已记录在experiment配置和Artifact metadata。

初始化使用作者n_jobs10，而早期串行验证使用n_jobs1，两者固定seed小样本结果不同；当前并行重复结果一致，不把旧串行checkpoint或源码指纹冒充当前生产设置。统一入口沿用项目 `torch.set_float32_matmul_precision("medium")`；同CUDA/精度下重复SID一致，不承诺CPU/CUDA编码数值一致。

## 上游输入准备

- 内容：沿用共同 embedding 生产协议，bundle 为 `{"keys": item_keys, "predictions": content_embeddings}`。真实 Beauty/Sports/Toys 共同内容均为1024维，正式命令显式 override `input_dim=1024`；768维仅为初次synthetic验证及组件通用默认。
- CF：必填 `cf_embedding_path`，同目录32维 bundle。必须与内容覆盖完全相同的 item keys；loader 按 key 对齐，不能按行号拼接。`cf_source` 必填，记录独立 CF 训练代码、训练 split、seed、模型/产物身份和维度。
- 官方仓库没有发布完整可复现的 CF teacher 训练实现，仅有导出片段和部分数据。新增 `src/recommendation/letter/cf_teacher.py` 独立实现32维SASRec teacher，不复用项目现有SASRec；对应 `letter_cf_train.sh` / `letter_cf_export.sh`。CF仍为显式上游输入，不把随机CF当正式输入。
- CF协议 `letter-cf-sasrec32-grid-v1`：history50、2blocks/1head/dropout0.5、逐位置正负BCE、完整training行负采样排除；Adam lr0.001/betas0.9,0.98，单卡batch128/FP32，50k step，每1000 step evaluation NDCG10选优。仅training参与梯度，不消费testing；该训练预算为GRID适配，不宣称复现作者未公布的teacher设置。导出全部目录的未归一化32维商品表；cold商品没有正交互监督，可能收到负采样梯度。
- 需要按 Beauty/Sports/Toys 准备各自的 CF，且只使用 training 交互学习 CF。新的 LETTER tokenizer 每个 seed 独立训练。正式推荐使用本实现生成的唯一四码 SID，不能用项目其他量化器的 SID 替代。
- checkpoint 带输入/目录内容 SHA256及模型结构身份；恢复时拒绝替换同shape目录或CF。身份数据为普通Python类型，支持 PyTorch `weights_only` 默认恢复。

## 从仓库根目录执行

九个槽位独立可复制的完整命令见 [九槽位命令](letter-issue-commands-20261003.md)，包含CF训练/导出、Tokenizer训练、SID导出、推荐训练、Testing和显式dry-run。各槽位使用固定内容版本和自己的seed，未来产物变量带shell守卫；真实训练后登记validation best与固定Artifact版本，不能使用验证用的有限训练产物。以下变量必须填写实际存在的路径/来源。所有脚本接受 `--notes="..."` 和 `--notes "..."`、显式 `--dry-run`、额外 Hydra override（最后覆盖默认值）。正式训练默认不启用dry-run；正式完整实验由用户手动开始。

```bash
DATASET=beauty
SEED=42
# CONTENT、CF、CF_SOURCE 必须指定真实输入及来源。
bash letter_tokenizer_train.sh --dataset "$DATASET" --seed "$SEED" \
  --embedding-path "$CONTENT" --cf-embedding-path "$CF" --cf-source "$CF_SOURCE" \
  --notes "LETTER native tokenizer; matched content and train-only CF" --dry-run input_dim=1024
```

移除 `--dry-run` 后执行正式 tokenizer 训练，选择 `val/collision_rate` 最小的 checkpoint；将其明确赋给 `TOKENIZER_BEST`，不要使用 last 代替选优结果。

```bash
bash letter_sid.sh --dataset "$DATASET" --seed "$SEED" \
  --embedding-path "$CONTENT" --cf-embedding-path "$CF" --cf-source "$CF_SOURCE" \
  --ckpt-path "$TOKENIZER_BEST" input_dim=1024
# SID 设置为导出运行的 predictions/merged_predictions_tensor.pt 或明确 W&B URI。
bash letter_train.sh --dataset "$DATASET" --seed "$SEED" \
  --data-dir "data/$DATASET" --semantic-id-path "$SID" \
  --notes "LETTER native SID; shared full-catalog protocol" --dry-run
```

移除 `--dry-run` 后执行正式推荐训练，选择 `val/ndcg@10` 最大 checkpoint，明确赋给 `LETTER_BEST`，然后执行Testing：

```bash
bash letter_inference.sh --dataset "$DATASET" --seed "$SEED" \
  --data-dir "data/$DATASET" --semantic-id-path "$SID" --ckpt-path "$LETTER_BEST" \
  --notes "Testing with validation-selected LETTER checkpoint"
```

默认推荐两卡；可显式指定 `--gpus 4,5 --nproc-per-node 2 --master-port 29521`。默认 tokenizer 一卡，拒绝多个进程。推荐验证使用 evaluation，推理使用 testing，不在训练后自动消费test。

正式矩阵为 Beauty/Sports/Toys × seed42/200/2026 共9次；每次绑定自己的CF、tokenizer checkpoint、四码SID、推荐best checkpoint和Testing输出。Beauty42对应 [BMX-60](https://linear.app/baymax104/issue/BMX-60/基线letter-beauty-seed42)，整体结果归 [BMX-130](https://linear.app/baymax104/issue/BMX-130/完成-letter-基线主表)。本次完成可运行准备与九槽位命令，尚未启动正式矩阵。未来产物由对应正式阶段产生，Testing等待真实validation best；不能以初始化探针SID或有限训练CF作为正式上游。

## 输出与验证边界

SID和推荐输出均为公共 keyed model output bundle；本地默认 `predictions/merged_predictions_tensor.pt`。推荐 predictions 为 `[users,10]` 原始 item keys。SID predictions 为 `[items,4]` 学习码。W&B分别发布semantic_id、推荐结果和validation选优checkpoint，配置记录所有结构参数和notes；可再生成fixture/缓存不发布。

初次GPU验证使用显式标记的synthetic fixture，后续可运行准备使用真实共同目录及有限训练CF/native SID；两者只验证运行正确性，不证明论文效果。详细记录分别见同目录 `letter-validation.md` 和 `letter-readiness-20261003.md`。
