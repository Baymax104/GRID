## ADDED Requirements

### Requirement: 冻结输入与开发边界
系统 SHALL 校验 E1 为正式 proxy_qualified、训练记录及目录/内容指纹一致，并仅使用 fit/selection 选择映射。

#### Scenario: 输入不一致
- **WHEN** E1 未通过或记录指纹不一致
- **THEN** 准备失败，不产生可训练 SID。

#### Scenario: 保留内部检查
- **WHEN** 改变 internal check 的目标
- **THEN** 已选映射保持不变，只有准备判定可能改变。

### Requirement: 等改动集合的受控置换
系统 SHALL 使用四不同首组的四 item 块，按冻结配对及几何阈值构造两臂；保持 tuple 集合和唯一性，两臂改动 item 集合完全相同。

#### Scenario: 成功匹配
- **WHEN** 接受一个块
- **THEN** 两臂各发生两个完整 SID 交换，每 item 首组改变，所有前缀目录占用不变。

### Requirement: 失败关闭输出
系统 SHALL 仅在支持、结构、代理转移门槛全部通过且非 dry-run 时发布三臂 keyed bundle。

#### Scenario: 信号未转移
- **WHEN** internal check 配对区间或 conditional NLL 不通过
- **THEN** 保存 no_go_proxy_transfer 与完整证据，但没有可训练 bundle。

### Requirement: 统一链路与匹配训练
系统 SHALL 经 src.main/Hydra/Trainer.test 准备，使用共享 writer；训练三臂采用同一 mask_ce、随机初始化、预算和checkpoint选择，运行前校验门禁和bundle指纹。

#### Scenario: 错误训练输入
- **WHEN** arm 与 SID 指纹不符或准备未通过
- **THEN** 在训练更新前报错。

#### Scenario: dry-run
- **WHEN** 根脚本传入 --dry-run
- **THEN** 统一入口禁用发布；notes两种语法和额外Hydra override均被保留。
