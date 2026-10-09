# M3 启动前只读核验（2026-10-08）

核验范围为本地源代码、node1 CPU 配置与输入文件元数据。本预检没有启动训练、推理、真实 Trainer 或 GPU probe，没有修改 src/configs、远端虚拟环境、研究状态或 Linear。

- 五臂统一使用 `copmrec_ablation_train.sh` → `copmrec_train.sh` → `liger_common.sh` → `uv run torchrun --nproc_per_node=2 -m src.main`。
- node1 五臂 Hydra compose 均通过：scratch/seed42、DDP2、global256（每卡128）、FP32、50k updates、每500步 raw dense validation、首次最高 val/ndcg@10 选 own-best、runtime source snapshot 开启。正式训练入口拒绝上游 recommendation checkpoint 初始化。
- A1/A4 gate 冻结且退出 optimizer，A2 残差权重冻结且使用单个 base-lr 参数组；其余臂保留两组参数。有效参数梯度覆盖及完整 checkpoint 恢复有准备阶段的 CPU 回归覆盖。本次未追加更新或 GPU smoke。
- source snapshot rank-zero 守卫同时检查 Lightning rank、torchrun RANK 和 distributed rank；非零 rank 不发布 W&B 源码 Artifact。
- node1 非交互 SSH 的 PATH 不含 uv。实际 tmux launcher 必须加入 `$HOME/.local/bin`；运行环境已有 `/home/weizhenyu/.local/bin/uv`，不需要安装依赖。

node1 当前304个受管源文件与本地逐文件 SHA256 完全匹配，来源记录 verified。source SHA256 为 `42f0d7c7dce0a09961e0c8c23f9ba35982abfb796d13952c3f17eaeba17c9f2e`。这里只读取本地同步生成的来源文件，未使用远端 Git 判断版本。

Beauty training/evaluation/testing 各175份 TFRecord.gz，合计525份，均登记大小和 SHA256。清单规范化 JSON SHA256 为 `46fbe99589b809ec8b7b9c21a815e4757c7a20d57664342ba8e525ecad8cb0c2`。175份 Testing 全部与正式主结果审计回执的路径、大小、SHA256 一致；主结果完成回执未包含 training/evaluation 历史字节清单，本次登记当前原始清单，不声称补齐历史来源。

| 输入 | Artifact | Digest | 文件 SHA256 | 形状 |
| -- | -- | -- | -- | -- |
| SID | rkmeans_inference_beauty-semantic-id:v0 | 19ace08283163fbe687287fbacaa3842 | c95072fc46b7a6a9359fda3c1225625cb2c3625a3be8b7683652ada17949489b | 12101 × 4 |
| Embedding | sem_embeds_inference-semantic-embedding:v5 | ab56af975eac589c27eed6094482cbd7 | 968b7fa491bd8ef6b2fe58942b45da10dbd788882ddf4e9a9de3b6cf1d0e11b1 | 12101 × 1024 |

两份缓存均与实时 W&B Artifact 文件 MD5 匹配，digest 与 Full 正式主结果采用值相同。SID 合法且全目录唯一，输入 keys 无重复。

详细回执为 `preflight-node1.json`、`preflight-local-match.json`；只读核验脚本为 `preflight-reader.py`。预检通过表示输入与启动协议满足当前检查，不是方法效果或训练完成的证据。
