# SASRec 基线实现与运行

2026-09-30 后续准备核验已完成：三个数据集固定 v0 目录与内容 keys 一致，node1 CUDA0 默认4 workers 的九个训练 dry-run 全部通过。正式目录指纹、运行环境及逐单元记录见 [准备核验报告](sasrec-readiness-20260930.md)。下方早期本地验证记录中的未确认事项，以后续核验报告为准；完整训练和 testing 仍待用户手动运行。

## 算法依据

实现依据作者 [kang205/SASRec](https://github.com/kang205/SASRec/tree/e3738967fddab206d6eeb4fda433e7a7034dd8b1)，固定 commit 为 `e3738967fddab206d6eeb4fda433e7a7034dd8b1`。原项目采用 Apache-2.0 许可。本实现用 PyTorch 重写公式，未引入第三方 SASRec 包；不以 pmixer 或 RecBole 的变体作为算法依据。

核心位于 `src/recommendation/sasrec/backbone.py`，保留如下计算：

- 商品 embedding 乘以 `sqrt(hidden_size)`，加固定槽位位置 embedding；左 padding 的有效商品 embedding 为零。
- attention 的 Q 来自 `LN(x)`，K/V 来自 `x`；保留因果 mask、官方 key/query mask、attention dropout，无 output projection，residual 加归一化 query 输入。
- FFN 为 hidden→hidden→hidden，使用 ReLU 和两次 dropout；residual 加 FFN 的归一化输入。每个 block 清零 padding，最后再做 LN，epsilon 为 `1e-8`。
- 每个有效位置用 tied item embedding 计算正负样本点积；目标为官方 `-log(sigmoid(pos)+1e-24)-log(1-sigmoid(neg)+1e-24)`，按有效位置平均。L2 为 raw item 与 position embedding 平方和的 `0.5*l2_emb` 倍。
- Adam 默认 lr=0.001、betas=(0.9,0.98)、eps=1e-8，无 weight decay/scheduler。默认历史50、hidden50、blocks2、heads1、dropout0.5、L2=0、训练 batch128。

测试使用独立 NumPy 公式逐层对照固定权重，以及因果性、padding、梯度与 loss 饱和边界检查。跨框架随机初始化和 dropout RNG 不保证逐位相同。

## GRID 数据与评价协议

`training/evaluation/testing` 沿用现有数据划分。训练使用每行 `sequence[:-1]` 输入与 `sequence[1:]` 逐位置标签，保留最后50个输入位置；负样本在固定全商品目录均匀抽取，排除完整 training 行，包括截断掉的历史与目标。不会读取 evaluation/testing 来构建排除集合。训练长度不足2的行跳过，评估无历史的行直接报错。

`item_catalog_path` 必须指向共享 model output bundle，SASRec 仅读 `keys`，不消费 SID 或内容向量。原始商品 key 排序后映射为模型 ID 1..N，模型 ID 0 专用于 padding，因此原始商品0仍可训练与推荐。正式比较必须使用与其他方法相同的冻结商品 key 集合；不能凭目录名称推断一致。

验证和测试以每行最后商品为目标，其前序历史为输入。对全目录点积排名，不移除历史商品，不删除冷启动目标，全部有效用户计入 Recall/NDCG 分母。按 score 降序、同分原始 key 升序排序；按目录块计算，默认 chunk4096，结果不随 chunk 大小改变。临时打分张量规模为 batch×chunk×hidden。预测输出为 `{"keys": 用户key, "predictions": 原始商品key[B,K]}`，使用共享 writer。

选模只看 evaluation 的 `val/ndcg@10`，不自动执行 testing。独立 inference 恢复指定 checkpoint 并评价 testing，默认由 callback 写 W&B summary。checkpoint 保存目录 SHA256 与算法结构身份，恢复时拒绝不一致目录或历史长度等结构配置；top_k/chunk 可调整。

GRID 的 streaming dataset、worker RNG、全目录评价和默认50000 optimizer steps 属于框架适配；作者原脚本的采样训练调度及100个候选评价不能直接作为本项目的比较协议。50000步只是默认预算，不代表已验证收敛。正式实验须冻结并记录训练预算、历史长度、目录、split、seed、评价协议，不能把本次逻辑验证当成性能结果。

## 手动命令

从仓库根目录运行。以下 Bash 命令中的目录 bundle 与 best checkpoint 由用户填写真实路径或 W&B 引用；不要用 final checkpoint 替代按验证指标选出的 best checkpoint。

```bash
CATALOG='填写与其他基线相同的冻结keys bundle路径'
bash ./sasrec_train.sh --data-dir data/beauty --dataset-name beauty \
  --item-catalog-path "$CATALOG" --devices '[0]' --seed 42 \
  --notes 'SASRec官方核心公式；共同split/catalog/full-catalog评价'

# 显式 smoke；不用于正式结果。
bash ./sasrec_train.sh --data-dir data/beauty --dataset-name beauty \
  --item-catalog-path "$CATALOG" --devices '[0]' --seed 42 --dry-run

BEST='填写上述训练run中selection=best的checkpoint路径或URI'
bash ./sasrec_inference.sh --data-dir data/beauty --dataset-name beauty \
  --item-catalog-path "$CATALOG" --checkpoint "$BEST" --devices '[0]' --seed 42 \
  --notes '恢复validation best；testing全目录评价'
```

W&B bundle URI 必须显式声明 `role=semantic_id` 及需要的 `file`，格式为 `wandb://entity/project/producer-run-id?role=semantic_id&file=merged_predictions_tensor.pt`；可用 `alias=vN` 固定 Artifact 版本。这里的 role 用于共享 reader 解析，SASRec 仍只读取 keys。checkpoint URI 则使用对应训练 run ID 与 `role=checkpoint`。不要写单引号内的 `${CATALOG}` 作为 Hydra 值。每个数据集使用对应目录 bundle，重复 seed 时保持其他协议不变。

脚本支持 `--notes=value`/`--notes value`，空 notes 不转发；额外 Hydra overrides 放在默认参数后，能够覆盖默认值。多卡例如 `--devices '[0,1]'` 自动使用 `uv run torchrun --nproc_per_node=2 -m src.main`，可指定 `--master-port`。如设置 `NPROC_PER_NODE`，必须与 devices 数量相符。seed 范围为1..4294967295。所有训练/预测均通过统一 `src.main` 入口。

PowerShell 可直接调用统一入口：

```powershell
$env:PYTHONUTF8='1'
uv run python -m src.main experiment=sasrec_train data_dir=data/beauty `
  dataset_name=beauty item_catalog_path='填写冻结bundle路径' devices=[0] seed=42
```

Windows 终端建议设置 `PYTHONUTF8=1`，避免 Rich 在退出时因 GBK 无法编码符号而报错。CUDA 验证环境必须安装 CUDA 版 PyTorch；此次使用独立临时环境，未改动项目依赖清单或原 `.venv`。

## 2026-09-30 逻辑验证记录

本地 RTX4060 Laptop、PyTorch2.9.1+cu128，独立环境位于 `%LOCALAPPDATA%/Temp/grid-sasrec-gpu-runtime`。原 `.venv` 为 CPU 版；未使用 node1、未启动完整实验、未发布 W&B 实验产物。

统一入口 dry-run 在 UTF-8 环境下成功退出。另一次训练完成5次 optimizer update，loss 为 1.390468、1.372082、1.402735、1.410077、1.386330；验证两批共16用户，保存 step5 checkpoint。最初退出因 GBK 编码失败，不能把该进程描述为正常退出。检查发现 Adam betas 被 OmegaConf ListConfig 序列化，配置已增加 `_convert_: all`，回归测试验证 optimizer state 可由 `torch.load(weights_only=True)` 读取。

仅对自己生成且来源可信的本地 checkpoint，将 betas 转为普通 tuple 并另存 `last-safe.ckpt`，保留原文件；随后从 step5 恢复、保持 max_steps5，正常退出且未增加训练步数。这是验证产物修复，生产 checkpoint loader 没有添加不安全反序列化回退。

恢复推理在 GPU 正常退出，读取 testing 两批共16用户，输出16×10原始商品 ID。核验用户 key 唯一且与 testing 目标一致、推荐 ID 属于目录且每行无重复；独立重算 Recall/NDCG 与 CSV 一致。此时命中指标均为0，不能用于评价基线效果。checkpoint 的 global_step 和所有 optimizer step 均为5，参数与 loss 有限。

验证目录来自本地 Beauty `items` 全量 keys，共12101项，SHA256 为 `5ff9b2543331cb866347349d3d66026bcfe1a5a4f501716ab09c3832ea06b43c`；尚未证明与正式比较的冻结 SID bundle 完全一致，正式运行必须指定共同 bundle。

本地产物保存在 `logs/sasrec-validation/`：`verification.json`、`five-step/checkpoints/last-safe.ckpt`、`restored-inference-fixed/bundle/merged_predictions_tensor.pt` 及 CSV/Hydra 配置。该目录用于复核逻辑，不是正式结果。默认4 workers、多卡、完整50000步与完整评价尚未进行实际运行验证。

最终聚焦回归71项通过，涵盖54项 SASRec 测试和共享 MetricEngine/MetricCallback/LocalPickleWriter 回归；新模块 Ruff check/format 及四个模块 OpenSpec strict 均通过。配置测试包含 Hydra compose 与安全 optimizer state 序列化；脚本测试包含 Bash 语法、引用、空值、错误输入、override 优先级和多卡命令构造。
