# CoPMRec 当前正式版本

当前版本为 **v2**。默认模型 `src.recommendation.copmrec.CoPMRec`，训练 `copmrec_train.sh`、推理 `copmrec_inference.sh`；所有阶段无历史排除，三项等权loss为SID CE、joint catalog CE、mixture NLL，无native view。

消融入口 `copmrec_ablation_train.sh` / `copmrec_ablation_inference.sh`，仅允许no_mixture、no_residual、no_joint_ce。机制入口 `copmrec_diagnosis.sh`：hits、residual、prefix；Trainer.test单卡、不更新权重。group继续使用paper_main/ablation/mechanism_copmrec_<dataset>，config/task/tag明确v2。

旧native/history控制、joint-only临时别名、v1评估、预算续训和混合终排入口已退出运行面；必要的共享组件收敛在copmrec包，LIGER原基线行为保留。历史源码和测试逐字备份见[运行源码归档](archive/copmrec-v0-v1-runtime-20261009/README.md)。旧v0/v1定义与指标只在文档保存，不提供旧版可执行入口。

[方法定义](../../research/docs/copmrec-v2-formal-definition-20261009.md)、[当前计划](../../research/docs/copmrec-v2-experiment-plan-20261009.md)、[历史结果](../../research/docs/copmrec-v0-v1-results-archive-20261009.md)、[迁移核验](evidence/copmrec-v2-consolidation-20261009/README.md)。运行面收敛迁移本身未启动实验；用户后续授权的M2九次训练与九次单卡Testing均已完成并通过独立审计，见[完成结果](evidence/copmrec-v2-m2-test-20261009/README.md)；[启动回执](evidence/copmrec-v2-m2-launch-20261009/README.md)保留启动时的事实。

