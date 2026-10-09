## ADDED Requirements

### Requirement: 匹配有界训练
系统 SHALL 从同一冻结 A 初始化同一 branch128，使用同一缓存顺序，比较 CE 和首次失败目标，固定200更新、有效batch32并记录可验证训练来源。

#### Scenario: 固定终点
- **WHEN** 用户运行正式目标资格训练
- **THEN** 只在200步发布checkpoint，不用validation选择、不续训，保留6400窗口曝光与初始头/顺序哈希

### Requirement: 可达首次失败监督
系统 SHALL 排除缓存补 gold 与padding来还原A实际topk，并仅在首次失败使用非gold第B竞争者margin1.0；全程存活样本用最终A beam内CE。

#### Scenario: 后缀不可达
- **WHEN** gold首次在第二层被剪掉
- **THEN** 监督第二层的真实竞争，不计第三层及以后的gold CE，所有层正则仍与对照相同

### Requirement: 有界配对评价
系统 SHALL 对同一512用户核对来源、冻结参数、off/zero复现后比较A及两臂，输出明确资格门槛。

#### Scenario: 目标未获资格
- **WHEN** 有效性通过但新目标任一预定效果条件未满足
- **THEN** 输出停止当前探针，不能声称稳定收益或自动扩大预算
