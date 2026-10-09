# LETTER 初次合成实现验证记录

验证时间：node1日志为2026-10-02 UTC，本地为2026-10-03 Asia/Shanghai。本记录保留初次合成验证的历史事实，仅证明独立实现和统一运行链路可工作。后续真实目录/CF/九槽位准备核验见 [LETTER可运行准备](letter-readiness-20261003.md)，不能把本记录的当时未就绪状态作为当前状态。

## 环境与输入

旧本地临时CUDA环境不可用，因此使用node1既有Python3.11、PyTorch2.9.1+cu128/CUDA12.8、Lightning2.6.5、Transformers5.14.1。未安装或重装PyTorch/CUDA。使用A100 GPU4/5验证推荐DDP，GPU6验证tokenizer；初期单卡检查使用GPU4。

所有运行从 `node1:/data3/weizhenyu/projects/GRID` 仓库根目录通过 `src.main`，最终复核使用新增根脚本。运行时 `UV_NO_SYNC=1`，使用已有虚拟环境。新增依赖为 `k-means-constrained`，实际版本0.8.0；其OR-Tools依赖使lock中的protobuf从7.35.1约束为6.33.6，Torch版本未变。

synthetic fixture保存在 `logs/letter_smoke_20261002/`，包含128个商品、16个用户、768维内容、32维随机CF，固定seed42。TFRecord使用公共reader要求的 `.tfrecord.gz` 后缀；每个split有两个分片。fixture的split用于链路校验，不作为无泄漏的正式数据证据。

为了适配小目录，tokenizer显式override codebook_size32、num_groups4、train batch16；推荐train/predict每卡batch4、worker0、timeout0。五步训练每5步验证，W&B logger/Artifact发布关闭。模型其余结构，包括四层码、latent32、完整MLP和T5骨干参数，使用组件默认。未执行正式256码本完整目录训练或正式9槽位实验。

## 检查结果

| 检查 | 结果 |
| --- | --- |
| LETTER聚焦测试及公共artifact回归 | 53 passed；4条第三方DeprecationWarning |
| Ruff | 新增LETTER实现和相关测试通过 |
| Hydra compose、脚本 | 四个experiment组合；Bash语法、notes两种形态、空值、非法输入、额外override、含等号checkpoint路径测试通过 |
| OpenSpec | 五个change均通过strict验证 |
| tokenizer dry-run | GPU，一次optimizer step，无业务结果写入 |
| tokenizer 5step | global_step5，所有optimizer state step5，参数有限、exp_avg非零 |
| tokenizer恢复/SID导出 | 默认weights_only恢复成功；输出128×4，全目录唯一，码值在[0,32) |
| 推荐dry-run | 两卡NCCL DDP，一次optimizer step |
| 推荐5step | 两卡DDP，global_step5，optimizer state step5，参数有限；val/user_count16 |
| 推荐恢复/预测 | 按val/ndcg@10选出的step5 checkpoint恢复；两卡合并16×10 CPU bundle |
| 输出独立核对 | 16个用户无缺漏/重复，所有商品属于目录，每位用户10个候选互异；逐用户Recall/NDCG与validation CSV一致 |

synthetic五步后的Recall/NDCG@5/@10均为0。tokenizer导出前最近邻碰撞率0.4140625，导出后的完整四码SID唯一。这些值只用于检查loss/评价/碰撞处理链路，没有效果解释。

## 最终复核产物

以下路径相对于node1仓库根目录，均实际存在：

- `logs/letter_smoke_20261002/tokenizer_release/checkpoints/step_step=000005.ckpt`
- `logs/letter_smoke_20261002/sid_release/predictions/merged_predictions_tensor.pt`
- `logs/letter_smoke_20261002/recommendation_release/checkpoints/step_step=000005.ckpt`
- `logs/letter_smoke_20261002/recommendation_release/csv/version_0/metrics.csv`
- `logs/letter_smoke_20261002/inference_release/predictions/merged_predictions_tensor.pt`
- `logs/letter_smoke_20261002/verification-release.json`

两份最终checkpoint的runtime source snapshot一致：

```text
source_sha256 = 24a6de0a1852aa31c46641704b50ca8083a79ee34701028526f3e67ee976147b
tokenizer checkpoint SHA256 = d97951e5a13b2b331cdddbbaa848a872769feead562045a3387c747068326a80
recommendation checkpoint SHA256 = a357778acb9b4d4e440e285d30e6cb6addcf7bdb11f1d20f4b6c0f8dc9e9a6b7
```

各训练目录 `metadata/source_snapshot/` 保留真实runtime字节归档、manifest和runtime.json，包含当时dirty/untracked源码；不依据远端Git推断来源。推荐训练实际消费 `sid_final` 目录中的native LETTER输出，独立断言确认其keys/codes与最终`sid_release`输出逐项完全一致。

最终训练和预测使用 `letter_tokenizer_train.sh`、`letter_sid.sh`、`letter_train.sh`、`letter_inference.sh`。resolved参数保存于各输出目录 `.hydra/config.yaml`，运行日志为相应 `*_release.stdout`。dry-run日志见 `tokenizer_final_dry.stdout`、`sid_final_dry.stdout`、`recommendation_dry_verified.stdout`、`inference_final_dry.stdout`。

## 已修复的集成问题

- tokenizer的paths.data_dir未覆盖公共配置插值，已明确设为当前目录。
- W&B logger target alias绕过dry-run关闭列表，已使用公共框架识别的完整target。
- checkpoint身份含Hydra ListConfig，已规范化为普通Python类型并添加weights_only回归。
- 早期fixture后缀不匹配，空stream出现无界循环；fixture已修正，LETTER数据适配对无可用训练样本显式报错。
- 直接命令中的checkpoint等号需要Hydra quoting；根脚本已正确处理并做回归。
- DDP pickle输出曾保留cuda:0/cuda:1；新LETTER模块输出CPU bundle，评价callback移回target device，两卡合并已复核。

早期失败与重试日志保留在同一smoke目录；未将其作为成功证据，已终止自己遗留的失败进程。

## 未验证范围

没有启动真实Beauty/Sports/Toys × 三seed正式训练，没有生成正式CF来源证明或正式推荐结果。本轮关闭W&B发布；真实run、W&B checkpoint/result lineage及正式Testing消费仍须在用户启动完整实验后审计。predict summary策略通过公共MetricCallback和模拟summary接口测试，GPU无W&B logger时只独立复算输出，不声称在线W&B指标已验证。

初次合成验证结束时BMX-58尚未Done、Linear状态未修改；后续可运行准备已另行核验并交付，见上述新记录。执行正式链路前仍需按 [LETTER运行文档](letter.md) 生产本单元正式32维CF、native四码SID和validation最佳checkpoint。
