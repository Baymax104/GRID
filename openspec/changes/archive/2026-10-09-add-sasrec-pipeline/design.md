## Context

项目统一 src.main 与 FileDataModule/MetricCallback/shared writer 已可装配前三个 SASRec 模块。用户授权本地 GPU dry-run 和5 step；完整实验仍手动。当前项目 .venv 为CPU torch，隔离临时 CUDA 环境使用相同 pinned torch/vision/audio，不修改项目依赖。

## Goals / Non-Goals

**Goals:** 可执行 train/inference 入口、明确协议、可审计选模和产物、真实有界验证。

**Non-Goals:** 不启动完整实验、不宣称5 step准确性、不自动使用 testing 选模、不增加新的实验方法变体。

## Decisions

- 组件参数归对应 model/data/trainer/logger/callback configs；experiment 只声明 defaults、人工路径、seed、运行元信息和高层关系。
- 官方默认 hidden=50、blocks=2、heads=1、dropout=.5、L=50、lr=.001、batch128、Adam beta2=.98，无scheduler。GRID streaming train使用 max_steps=50000 和每1000 step validation；明确这是训练预算适配，不声称作者原训练日程或收敛已验证。
- 保留固定全目录、全部用户和不屏蔽历史，checkpoint只由 evaluation的val/ndcg@10选取。testing只用于独立推理，训练不自动test。
- item_catalog_path 明确传给共享 bundle loader，仅读keys；允许复用相同 SID Artifact 的目录，URI须显式role/file。
- train/inference 根脚本复用一个小公共参数解析器，未知flag报错，额外Hydra override置于默认值之后；多卡torchrun从devices推导进程数并核验显式NPROC。
- 正式推理用W&B共享writer，另提供关闭发布后选用LocalPickleWriter的override。prediction callback的同一test metrics与训练validation保持一致。
- 有界验证不在单元测试内运行Trainer；通过 src.main 显式 overrides执行。检查5 step checkpoint的global_step、finite参数和optimizer step，并实际恢复推理bundle及重算metrics。

## Risks / Trade-offs

- CUDA wheel获取与项目索引冲突 → 隔离环境使用官方明确wheel URL，不改项目 .venv。
- 本地数据可能是开发样本 → 记录规模及用途，不写成正式Beauty指标。
- 正式性能未验证 → 用户手动完整训练后审计W&B，当前只承诺逻辑与协议。
