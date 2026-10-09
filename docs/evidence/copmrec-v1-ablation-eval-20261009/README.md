# CoPMRec v1 消融重新评价（2026-10-09）

用户明确要求按 v1 重新进行消融评估。父任务 BMX-150；BMX-151～155 复用五臂原 own-best，各一次无历史排除单卡 Testing；BMX-156 为六份 v1 bundle 的 M1 分析（model-forward=0）。Beauty/training seed42；0新训练/0独立Validation/5Testing/1分析，Full hc8oct43 直接复用。

本批次已完整运行并独立核验，BMX-150及六个子issue已登记Done/有效。五次Testing与M1均exit0；四指标最大复算误差6.6e-17。M1六臂134178 user-variant/114slice/608配对CI/48hit count/132固定案例；三项NDCG贡献可加、指定A4−A1/A5−A3与全部切片/文件身份独立核验。

- [六臂原值、主要指标差与百分比](tables/six-arm-metrics.md)、[机器实证汇总](empirical-summary.json)。
- [完整研究报告](../../../../research/docs/copmrec-v1-ablation-evaluation-20261009.md)。
- `testing-audit-summary.json` / `testing-audit-BMX-151.json`～`testing-audit-BMX-155.json`：不可变来源、run config、输入/源码archive/产物身份、合法性、指标及paired CI。
- `m1-independent-audit.json`：全量用户级统计、608区间、固定案例、可加分解与14个文件核验。
- `tables/`：六臂四指标、全部变体−Full原差/百分比/CI、指定对照和NDCG贡献CSV，附SHA manifest。
- `local-verification.json` / `completion-verification.json`：CPU恢复/评分/协议、脚本/配置、OpenSpec与原研究证据保持检查。
- `linear-completion-writes.json`及`linear-final-readback.json`：线上登记与独立回读。

scope已执行完毕，不自动增加训练/Testing/诊断预算。原v0结果保持，当前新增证据仍只覆盖Beauty/seed42。

源码与配置：新增薄 v1 消融推理/M1 experiment 和根脚本，原 v0 默认及五臂 checkpoint 契约保留；CPU 五臂恢复、参数/原评分相同、错误来源拒绝、M1零forward/历史规则与合法唯一检查、脚本 quoting/override/dry-run、Hydra compose 已验证；strict OpenSpec通过。

- `testing-launch-specs.json` / `BMX-151-command.sh`～`BMX-155-command.sh`：精确 URI/SHA、原训练与选点、运行条件、GPU→local、tmux及命令。
- `m1-launch-spec.json` / `BMX-156-command.sh`：Full与五臂 bundle URI、零forward分析命令。
- `prepared.json`：有界成本与源码身份。
- `linear-initial.json`：统一十章模板的初始issue回执。
- `research-state-before.yaml` / `current-plan-before.md`：授权执行前的原始状态快照。

结果只报告原值、带符号差、N、区间、预定切片与缺失项，不提供好坏/积极消极分类。独立核验来源、完整用户/标签、合法唯一Top10和四指标/统计后登记完成；原v0 M3指标与250k训练记录保持。
