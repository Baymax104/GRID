## ADDED Requirements

### Requirement: 有界冻结A审计
系统 SHALL 通过src.main与Trainer.predict，对已加载的原A checkpoint执行单进程evaluation审计，默认128用户、beam10；SHALL拒绝training、testing、二层交互及在线评分干预。

#### Scenario: 错误配置保护
- **WHEN** 没有加载checkpoint、处于train模式或非A条件
- **THEN** 审计报错，不输出可被当作有效审计的证据

### Requirement: 完整目录与真实搜索比较
系统 SHALL 对每个合法item按自己的完整SID路径计算归一化概率，并调用原A实际beam；SHALL保存精确排名、TopK、beam输出及前缀生存，保证标签不影响预测选择。

#### Scenario: 分块和真值变化
- **WHEN** 改变精确评分chunk或真值标签
- **THEN** 目录概率数值一致且beam选择不变，标签相关统计允许变化

### Requirement: 数值和证据验证
系统 SHALL 保存checkpoint/catalog/input身份、完整概率、稳定排名和近同分上下界；SHALL从这些值验证搜索遗漏、评分失败和不确定类别，不能把精确相关性表现当理论上界。

审计 SHALL 将编码、目录评分和实际beam调用统一置于highest float32 matmul精度，退出或异常时恢复调用方设置，并将有效精度写入证据。评分一致性阈值保持atol=1e-4、rtol=0；失败时SHALL输出最大误差、用户/item身份、两条logp、chunk和设备。

#### Scenario: 入口启用medium矩阵乘
- **WHEN** src.main在审计前启用了medium精度
- **THEN** 审计三条计算路径均使用highest，结束或失败后恢复medium，仍拒绝超过原阈值的评分差异

#### Scenario: 有并列排名
- **WHEN** 目标的数值容差排名区间跨越K
- **THEN** beam未命中被记为边界不确定，不武断归为搜索错误

### Requirement: 可手动执行的操作契约
根脚本 SHALL要求data-dir，支持notes两种语法、seed、dry-run及额外override最后生效；SHALL默认一个GPU，通过共享writer和lineage callback发布审计；不自动运行G1。

#### Scenario: 数据集样本不足
- **WHEN** evaluation唯一用户数少于请求数或出现重复key
- **THEN** deterministic sampler报错而不静默输出少量样本
