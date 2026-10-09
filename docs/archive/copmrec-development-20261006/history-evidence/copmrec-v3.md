# CoPMRec v3：首层 Max、后续 Mass

日期：2026-10-04。用户授权从真实 v0（BMX-116）派生 v3，并要求训练配置与 v0 相同。当前实现将研究重点从 content 终排转向候选前缀搜索；此前 content bad case 分析保留为历史证据，v2 已关闭的迭代和既有预算不恢复。

当前进展：用户训练run [rbha00vx](https://wandb.ai/baymaxam/GRID/runs/rbha00vx) 已finished并通过配置、源码及best checkpoint核验；best为step43500，记录validation NDCG@10=0.045608、Recall@10=0.089815。固定best的 [单卡Testing命令](copmrec-v3-rbha00vx-testing.md) 已交付，testing效果待运行后核验。下方实现交付时的“尚待训练”与run数说明保留其历史含义。

当前效果：用户单卡推理run zy8946l2已通过独立Testing复算，Recall10=0.069892、NDCG10=0.035701，@10点估计低于真实v0/LIGER dense、配对区间跨零；首层dense Top10遗漏为0，但2–4层仍遗漏114。当前v3暂不晋级，见 [完整结果](copmrec-v3-testing-zy8946l2-result.md)。此前“待Testing”保留为历史阶段记录。

## 方法与训练

模型为 `src.recommendation.liger.root_max_mass.RootMaxMassCoPMRec`，继承 `JointMixtureLiger`。对用户u、父前缀p、合法子节点t及其后代商品D(pt)，内容logit为c(u,i)：

```text
第1层：V(u,pt) = max_{i∈D(pt)} c(u,i)
第2–4层：V(u,pt) = logsumexp_{i∈D(pt)} c(u,i)
q(t|u,p) = softmax_合法子节点(V(u,pt))
p_mix(t|u,p) = (1-alpha)*p_gen_合法(t|u,p) + alpha*q(t|u,p)
alpha = sigmoid(全局可学习bias)
L = SID CE + content CE + teacher forcing mixed NLL
```

训练与推理共用 `root_max_mass`。每层在本层合法兄弟节点内归一化；后续层不使用根层max作为mass分母。首层Max改变路径内容分布，不能宣称完整路径相乘仍等于原商品content softmax。基础content CE仍作用于原内容评分；全部推荐参数从零共同训练，无teacher、冻结阶段、额外head或ranking loss。

候选采用一次beam20加全部cold，终排沿用content分数。改变首层会改变后续frontier和累计分数，Max与Mass已有局部优势不能直接相加；v3效果尚待正式训练和匹配评价。

## 与真实v0相同的运行配置

v3 experiment直接继承 `copmrec_v0_train/inference`，model继承 `copmrec_v0`。除聚合与模型target外，仅运行版本、group、task/Artifact名称和protocol标签变化。

| 配置项 | v0 / v3 |
|---|---|
| 数据 | FileDataModule；完整training、全evaluation验证、testing推理 |
| 验证评分与best | dense content，val/ndcg@10，max |
| 最终推理 | hybrid候选、content终排 |
| 参数与初始化 | 相同state_dict键/参数数量；随机初始化；全部推荐参数可训练 |
| 主干 | d128、6层、6头、d_kv64、d_ff1024 |
| 训练更新 | 50000；无默认checkpoint恢复 |
| 每卡train/val/predict batch | 128 / 32 / 32 |
| 双卡与累积 | 原ddp、累积1；有效训练batch256 |
| 精度与梯度裁剪 | 32-true / 1.0 |
| 验证与训练日志间隔 | 500 / 50微批 |
| AdamW | lr0.0003，weight_decay0.035 |
| Scheduler | warmup2500，cosine总长50000 |
| beam / TopK / 历史 | 20 / 10 / 20商品 |
| dropout / input / projection | 0.2 / 0.5 / 0.2 |
| 内容temperature | 0.07 |

沿用v0的原DDP及指标配置，包括已有分布式执行方式。本版没有继承v2的额外DDP/指标同步调整；CPU双进程检查不等于完整GPU训练吞吐或验证指标审计。

## 从零双卡训练命令

在node1仓库根目录执行。物理GPU2、3通过 `CUDA_VISIBLE_DEVICES=2,3` 映射为Trainer逻辑 `[0,1]`。SID及embedding引用来自真实v0 Beauty协议；只预生成这两项输入，不预训练或恢复推荐模型。端口可根据运行占用修改。

```bash
cd /data3/weizhenyu/projects/GRID
export PATH="$HOME/.local/bin:$PATH"

CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 bash copmrec_v3_train.sh \
  --dataset beauty \
  --data-dir data/beauty \
  --semantic-id-path 'wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt' \
  --embedding-path 'wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt' \
  --devices '[0,1]' \
  --seed 42 \
  --master-port 29547 \
  --notes 'CoPMRec v3：由真实v0派生，首层Max后续Mass；三项基础loss等权，全参数从零训练；训练配置与v0相同，dense验证选best' \
  ckpt_path=null
```

脚本通过 `uv run torchrun --nproc_per_node=2 -m src.main experiment=copmrec_v3_train` 启动。默认不dry-run；显式 `--dry-run`才使用统一入口的有界逻辑。支持两种notes形式与末尾Hydra override，用户override优先。正式运行沿用resolved配置、上游Artifact lineage及实际运行字节源码快照。

## 推理与恢复

提供 `copmrec_v3_inference.sh`，训练完成后先核验validation-selected best，再填入精确checkpoint URI。v3 checkpoint记录版本和聚合/目标契约，拒绝v0/v1/v2或缺少契约的恢复。默认不提供v0 weights-only预训练入口，以保持本次从零协议。

candidate/path trace记录 `copmrec_version=v3`、`prefix_aggregation=root_max_mass`、逐层 `[max,mass,mass,mass]`及checkpoint alpha，继续使用共享writer和keyed model output bundle。

## 验证与研究边界

本地120项聚焦检查通过、1项Linux Gloo在Windows跳过；node1另13项CPU检查通过，包含原v0默认buffer广播的两进程连续更新与参数同步。v0/v1/v1.1/v2八份resolved有效配置完全不变，21个已有专属运行文件字节不变；共享代码仅增加固定聚合与训练装配扩展点。v3与v0完整配置差异仅为模型target、聚合与版本/日志/产物身份，25个运行文件本地/node1 hash匹配，四个配置及训练/推理命令替身核验通过。Mutagen三会话Watching、无conflict，Ruff与OpenSpec strict通过。

验证记录见 [verification.json](evidence/copmrec-v3-20261004/verification.json)，旧版本基线见 [before.json](evidence/copmrec-v3-20261004/before.json)。实现检查覆盖独立概率枚举、梯度、teacher forcing与逐步解码一致、checkpoint/optimizer恢复、真实HF beam/trace、标签独立、脚本与Hydra配置匹配，以及Linux两进程最小输入更新。

用户授权本次实施与训练命令交付，完整运行仍由用户手动开始。Agent完整训练/推理run数为0，不追加或重置过去关闭预算。先比较v3与真实v0的候选搜索损益和最终Recall/NDCG，再判断相对LIGER dense的收益；覆盖改善不能单独代替净收益。testing已经用于开发分析，不能包装为未来完全独立确认。
