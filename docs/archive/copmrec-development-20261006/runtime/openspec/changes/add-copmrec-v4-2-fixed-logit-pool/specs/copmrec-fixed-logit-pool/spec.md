## ADDED Requirements

### Requirement: 两个成员独立编码的完整目录固定 logit pooling

系统 SHALL 对相同真实用户历史分别运行 source 与 control 自身的 encoder / query / dense_logits，在相同完整目录逐行固定计算 `0.5 * source_logits + 0.5 * control_logits` 后稳定降序取 Top10 完整 SID；MUST 不依赖 labels、已有 Top10 或候选并集分数。

#### Scenario: 完整目录存在成员 Top10 之外的融合结果
- **WHEN** 两成员各自产生相同目录尺寸的 dense logits
- **THEN** 系统逐项平均完整 logits 后统一排序，不先截断、不平均 query / item embedding 或模型参数

#### Scenario: 同分与无标签推理
- **WHEN** 平均分相同，或同一 model input 配有不同 labels
- **THEN** ties 按固定 catalog row 稳定处理，标签变化不改变 keys / predictions

### Requirement: 严格双来源和原模型正常恢复

系统 MUST 通过公共 catalog / checkpoint loader 读取一个 catalog 与两个原始 checkpoints；四个必填 expected reference / SHA 与 loader 注入实际 identity 一致。生产配置及 preflight SHALL 固定 nj9elah1 best6000 SHA7c52228a665b593b50e45e8aba039f3f4aa4a10ca134d9345c59c59fb142f055 和 l3zyr91b best6000 SHA4bea4d5771c4f056ad6e0cd69d1204a3220b639868803cbc19aa3332ecf0a3c9，不替换 pair。

#### Scenario: 合法 source 与 frozen-bias control
- **WHEN** 输入原 v4 source 和 v4.1 scale0 control 的 best6000
- **THEN** 各自调用原类 on_load_checkpoint / strict load，catalog 全 buffers 一致、原 v0 来源完全一致、control 的 v4 warmstart 指向 source；不桥接版本、不恢复训练状态

#### Scenario: 来源或结构不匹配
- **WHEN** reference / SHA / version / step / catalog / 原 v0 链不一致，control 含非零 bias，cold residual 非零或 state 非有限
- **THEN** 构造或 preflight 失败，不静默重排目录、补参数或替换模型

### Requirement: 仅单卡推理且保留统一 pipeline 输出

系统 SHALL 经 src.main / Hydra、标准 ModelOutput 与 common writer 输出恰 keys / predictions bundle，保留两个 checkpoint 的公共 registry / lineage 和 pool_contract metadata；MUST 仅单进程单卡、dense full-catalog 固定评分，无 wrapper 独立 checkpoint。

#### Scenario: 统一脚本透传
- **WHEN** 根推理脚本收到 notes、dry-run、quoted URI 和允许的额外 Hydra override
- **THEN** 参数经统一入口传递，顶层 ckpt_path 为 null；两来源实际身份与冻结配置仍通过校验

#### Scenario: 非法训练或多进程
- **WHEN** 调用 fit / train(True) / training_step / configure_optimizers、独立 wrapper checkpoint save/load，或独立推理 world_size 大于1
- **THEN** 系统明确拒绝；不启动新训练或恢复旧 optimizer

### Requirement: 固定一次验证与累计预算

研究执行 MUST 保留已关闭 v4 / v4.1 的5次训练 / 30000 steps，新增训练0；固定唯一 source/control 与0.5/0.5，只做一次完整 Evaluation，不扫描 weight / temperature / normalization / pair / checkpoint / bias / LR。

#### Scenario: 正负或不确定结果
- **WHEN** 完整 Evaluation 输出经 raw labels / lineage / source 独立审计后产生结果
- **THEN** 复算对 source/control 和固定 LIGER 的指标与 paired CI，报告 fixed639 / other21724、warm/cold、固定 key 分组及 new/lost/shared；未达整体门槛关闭问题，不自动扩预算

### Requirement: 总体晋级与组件增量分开判断

系统 SHALL 以同 split LIGER5azn5vm0 的 R10≥.09877029021151008 / N10≥.0511173919307933 且两项相对 LIGER paired delta CI 下界为正作为唯一 pool 进入一次 Testing 的条件；MUST 不额外要求 pool-control 增量 CI 为正。Testing 基线与阈值保持042139al R10≥.07855386128873586 / N10≥.04002978116676896，累计最多3次、此前1次、本阶段最多1次。

#### Scenario: 总体达标而 pooling 增量不确定
- **WHEN** pool 通过上述整体 Validation 条件，但相对 control 的增量 CI 跨0
- **THEN** 可按预承诺进行唯一固定 Testing，同时报告增量不确定，不称 pooling 增量已确认，不用 Testing 重新选择规则

#### Scenario: 只通过一项或只显示互补
- **WHEN** Validation 只有一项达到10%、只显示成员并集命中或曝光减少
- **THEN** 不进入 Testing、不降低阈值、不改 pair 或权重，保留目标未完成结论
