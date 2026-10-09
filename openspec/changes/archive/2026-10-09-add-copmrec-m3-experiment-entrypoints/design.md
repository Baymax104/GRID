## Context

正式 v5.3 的损失和恢复契约写死。M3 使用 Beauty/seed42、五个训练变体和三个诊断证据包；完整运行仍须用户启动。

## Goals / Non-Goals

提供真实根目录命令、可 compose 配置、独立消融 checkpoint 身份与可复核诊断产物。此次不启动真实训练或推理，不改变 Full 发布契约、既有主结果与消融预算。

## Decisions

- 消融继承正式数值实现，覆盖损失组合与变体契约；A1/A4 冻结 gate，A2 冻结零残差。复用 scratch 的预算/状态验证，优化器组通过可覆盖的参数与峰值 lr 接口表达，Full 仍是原两组。
- 诊断包装来源模型并通过 data/components 的 checkpoint helper 校验 SHA、恢复契约和 strict state；bundle 分析使用已有 keyed reader，不新增下载逻辑。
- 诊断使用 FileDataModule 的 test 阶段读取相同 testing 数据。逐用户结果在内存累积，epoch 末由共享 structured writer 一次写入 JSON/CSV 和 keys/predictions bundle，避免各 batch 覆盖。
- W&B group 遵循 paper_<阶段>_<方法>_<数据集>，使用 paper_ablation_copmrec_beauty / paper_mechanism_copmrec_beauty；变体与阶段另记 config/notes/tags。新 checkpoint 先用明确占位变量，实际训练后按 own-best 审计补齐，不能用 Full 替代。

## Risks / Trade-offs

- 诊断需要收集完整用户级表 → 单卡、按批处理，保留紧凑观测列而非全目录 logits，记录用户全集和来源。
- Windows 无实际 NCCL 双卡 → CPU 梯度/完整恢复/静态参数覆盖与脚本配置验证；正式 Linux GPU 运行未经验证，不能声称已运行。
- 用户 bootstrap 不体现初始化波动 → 单 seed 的范围写入 metadata 和 issue。

## 实际执行修复（2026-10-08）

用户后续授权开始现有 M3 计划。真实 Hydra 默认将 input_references 传为嵌套 DictConfig，不能直接放入 summary JSON；诊断 constructor 使用 OmegaConf.to_container(resolve=True) 将来源引用转为普通 dict/list。共享分析 writer 在 constructor 同样规范化 metadata，覆盖 manifest 与 W&B Artifact 发布边界。测量公式、来源 checkpoint、训练计算和预算保持；回归覆盖嵌套 ListConfig 与插值。失败 attempt 保留，以新 run / tmux 完整重跑，无完整 manifest 的 run 不计交付。
