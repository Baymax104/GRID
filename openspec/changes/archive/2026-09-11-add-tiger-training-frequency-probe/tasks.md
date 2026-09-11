## 1. 训练统计

- [x] 1.1 实现期望目标计数、training-only 校验与来源摘要。
- [x] 1.2 测试抽样公式、未知 keys、零频、空输入和 RNG 不变性。

## 2. 独立训练探针

- [x] 2.1 实现条件分支权重和独立 training_step，保持评估与 state_dict 兼容。
- [x] 2.2 实现严格权重初始化、恢复冲突拒绝与 checkpoint 审计 metadata。
- [x] 2.3 验证 loss/梯度/更新等价性、权重边界、父前缀区分和原 Tiger 加载。

## 3. 配置与人工运行

- [x] 3.1 新增独立实验配置与根脚本，固定预算、last checkpoint、testing 关闭。
- [x] 3.2 验证 Hydra compose、脚本语法、参数 quoting/错误/末尾 overrides。
- [x] 3.3 完成配对运行手册、来源清单和方向验证门槛。

## 4. 验证

- [x] 4.1 运行聚焦测试、相关回归、Ruff、diff 检查与 OpenSpec strict，不启动完整实验。

验证记录（2026-09-12）：新增聚焦测试 36 项通过；完整 pytest 467 passed、4 个上游弃用 warning；Ruff 与 OpenSpec strict 通过。所有实现均为新增文件，baseline 无 tracked diff。真实数据/GPU 训练由用户在服务器完成，配对结果记录于 `results.md`。

## 5. 配置继承回归修复

- [x] 5.1 修复 model/callback defaults 对带 `_global_` 配置的继承，显式使用 `tiger_train@_here_`，避免父配置泄漏到顶层。
- [x] 5.2 检查继承的构造参数、optimizer、metrics 与 callback targets，并用 tiny 本地 checkpoint 实际实例化两种 arm 的 model/optimizer/callback；不执行 Trainer 或真实实验。

修复验证：配置与脚本测试 16 passed，Ruff 与 OpenSpec strict 通过。此前 compose 测试仅检查新增字段，未验证父配置完整性，因此曾漏检本问题；新测试覆盖该缺口。服务器需同步两个修正后的 YAML 后重跑原命令。

## 6. 三组方向验证与结论

- [x] 6.1 核验 seed 42/2024/2025 的六个训练 checkpoint、trace 与 diagnosis lineage，确认配对身份和固定预算。
- [x] 6.2 汇总 Overall/Head/Mid/Tail 命中迁移、NDCG 与第 2–3 层 teacher-forcing 证据，并按预声明门槛判断。
- [x] 6.3 形成 `results.md`；记录 Tail 微小一致收益、Mid 主收益、机制未稳定重复及 `stop_current_variant` 决策。

归档前验证：完整 pytest 467 passed、4 个上游弃用 warning；Ruff 与 OpenSpec strict 通过。
