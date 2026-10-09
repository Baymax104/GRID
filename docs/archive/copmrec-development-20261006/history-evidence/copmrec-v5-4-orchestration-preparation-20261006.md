# CoPMRec v5.4：专用编排与审计准备

## 真实授权与固定范围

用户在 2026-10-06T11:01:32.778Z 明确回复“批准运行 v5.4”。这条直接回复承接此前固定方案：新增 1 次随机起点连续 50000 更新训练＋1 次单卡完整 Validation；累计上限 5 次训练／250000 更新／5 次完整 Validation。新增 Testing、seed43 pair 和扫描均为 0。

[真实批准输入](evidence/copmrec-unified-catalog-only-residual-50k-20261006/explicit-authorization-input.json)绑定当前 thread 的原始用户消息、session context、此前 v5.4 问题以及固定 proposal。直接回复不伪造成 async 选项答复，也不要求用户再次点击。原问题记录仅证明问题已发出，不代表批准；旧 v5.3 批准已消费，不能授权新实验。

本记录形成时专用本地模板已验证，额度登记和生产预检仍须实际执行；尚无 v5.4 正式训练、checkpoint 或完整推荐结果。本地模型实现与 144 项核心／配置检查见[原准备快照](copmrec-unified-catalog-only-residual-50k-research.md)，其“等待答复”描述保留当时状态。

## 已补齐的运行绑定

原 v5.3 编排绑定第4次实验、408源码和 `native_view_ce`。v5.4 与它同为21键 checkpoint 合同，却改为 `residual_placement`；字段数相同不能证明兼容。

- 新 `common.txt` 逐字节核实际直接回复及其问题／thread／proposal来源；只有专用批准输入和新注册均存在时，才允许远端入口。固定预算严格区分 int、float 和 bool。
- 新训练 driver 固定 v5.4 target、根脚本和414源码，保留 Bash argv 捕获、完整 Hydra resolve、真实CPU属性回填、双rank smoke及唯一job／PID／exit marker。真实预检和启动不使用 fixture 成功凭据。
- 新推理 driver 只消费未来 v5.4 own-best checkpoint 原始 URI／SHA／producer／step，使用单GPU／local0／单进程完整 Validation，拒绝 Testing／43。
- 新训练和推理审计核 placement exact3、两处 catalog sharing、residual protocol、输入和实际source；完整50k预算与保存至own-best的完整状态分别证明。
- v5.2描述性参考独立按自身20键及原 history+catalog sharing 核验，不能从v5.4删一字段后改写参考契约。
- 登记模板先核实际批准，再保存原4／200k／4闭合ledger及state快照，只写新阶段登记；原账本和原state主组／closed tail保持。

固定效果门禁仍是同seed native的R10、N10各相对至少8%且两paired绝对95% CI下界正。固定v5.2只描述增量；没有新增parent硬门槛，也没有把query冻结或history residual有害作为已证事实。

## 本地实际检查

| 实施者及范围 | 结果 |
|---|---|
| evidence：专用编排／授权守卫 | 51 passed，0.57s；Ruff check／format check通过 |
| architecture：训练／推理审计守卫 | 59 passed，1.40s；Ruff check／format check通过 |
| root：登记模板 | 4 passed，0.06s；Ruff check／format check通过 |
| 两个driver和两个auditor的纯self-test | 全部exit0；仅fixture／静态编译／Bash stub／轻量compose |
| 第三方独立代码和接口review | 最终只读审阅通过；报告绑定实际SHA，不重复已通过的测试 |
| OpenSpec strict | 通过；正式阶段的实际任务另行记录 |

这些114项检查不使用真实数据、生产模型或Trainer，不执行正式训练／推理。root另外做过一次node1只读资源查询：当时物理GPU5、6、7空闲；拟训练5、6→local[0,1]、完整Validation单卡，实际启动前重新检查。

本地runtime仍414文件，source SHA `ae36e28d86d5a4c47c63d4f57b7a7753e2fe483168e9e1be62c414b47c6687c6`；本轮只增加docs helper／测试／记录，旧408及之前v5.4模型运行字节保持。

## 执行顺序与证据

从仓库根目录通过 `uv run` 执行本次docs helper，模型实际训练和推理均由根脚本进入 `src.main`；helper不另造模型运行入口。

```powershell
uv run --no-sync python docs/evidence/copmrec-unified-catalog-only-residual-50k-20261006/register-authorized-extension.txt
uv run --no-sync python docs/evidence/copmrec-unified-catalog-only-residual-50k-20261006/stage-driver.txt prepare candidate42 5,6
```

之后使用官方 `mutagen_sync.ps1 flush/status` 并记录三个Watching、无conflict，再做实际CPU空optimizer预检、双rank一步smoke和真实实现凭据，最后只启动一次正式50k。训练终态／100raw点／own-best／source及输入审计通过后，才准备唯一单卡完整Validation并独立核175文件／22363用户。任何已有job只观察原PID／handle，不因一次观察超时而重启。

正式启动需要实际CPU与smoke凭据；本文及[本地编排准备记录](evidence/copmrec-unified-catalog-only-residual-50k-20261006/local-orchestration-preparation.json)不能替代它们。原生LIGER历史source缺口、同更新预算不等FLOPs、单seed不等可复现整体目标等边界保持。
