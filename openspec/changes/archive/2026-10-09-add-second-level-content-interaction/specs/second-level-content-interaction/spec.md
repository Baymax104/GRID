## ADDED Requirements

### Requirement: 可复现的目录交互分解

系统 SHALL 对固定合法目录按物品等权拟合父码与第二层码的加性表示，依design使用float64、连通分量gauge及明确收敛条件；记录输入hash与拟合残差。系统 MUST 使用现有公共Artifact读取设施。

#### Scenario: 不均衡与不连通目录
- **WHEN** 输入包含不均衡pair占用数或多个连通分量
- **THEN** 拟合遵守物品等权，两侧组残差均值达到1e−10阈值，并在每个分量固定gauge；失败时明确报错

### Requirement: 从A开始的单标量分支

系统 SHALL 使用固定全局RMS校准的二层残差与一个共享tanh标量，标量初值为零；所有原A参数初始化与随机流保持一致，原A模式不新增状态。

#### Scenario: 零起点与梯度
- **WHEN** 对同seed的候选与A做eval前向和构造性梯度测试
- **THEN** 候选初始输出与A一致，非退化构造下标量可获得非零梯度，优化器包含且仅包含一次新增标量

### Requirement: 可归因的固定输入对照

系统 SHALL 支持off、interaction、additive、shuffled；后三者均只增加一个标量并使用相同能量校准。shuffled MUST 使用局部随机流并在置乱后重投影去除加性成分，记录可复算身份。

#### Scenario: 置乱有效性
- **WHEN** 构造shuffled对照
- **THEN** 全局训练随机流不变，报告移动比例及重投影后buffer hash；退化或与原分支相同的对照被拒绝

### Requirement: 因果一致的训练与搜索

系统 MUST 在encoder、teacher forcing、CGBS实际beam三处遵守相同二层表示规则；只使用已知item历史及已生成目标前缀，padding和SEP不注入。

#### Scenario: 未来目标变化
- **WHEN** 固定encoder状态并修改decoder尚未可见的目标suffix
- **THEN** 更早位置logit不变，第二码注入不影响该码自身的预测

#### Scenario: 同前缀评分一致
- **WHEN** eval下同一合法前缀分别走teacher forcing与beam评分
- **THEN** 对应局部logit与合法归一化分数在既定数值容差内一致，beam预算及非法候选处理不变

### Requirement: 持久化与历史兼容

系统 SHALL 将新增buffer、冻结尺度和构造身份纳入新checkpoint，推理恢复保存值并核验身份；off MUST 保持历史A契约及加载能力。

#### Scenario: 保存后恢复
- **WHEN** 已训练候选保存并由推理配置重新加载
- **THEN** 不按已训练SID表重算RMS尺度，输出与保存前一致，目录或模式不匹配时拒绝加载

### Requirement: 有限实验与入口

系统 SHALL 通过src.main及根脚本执行显式单条件实验，支持必填data-dir、seed、dry-run、两种notes语法及最后优先的Hydra override；默认不启动矩阵或训练后testing。完整实验 MUST 由用户手动启动。

#### Scenario: 单条件手动训练
- **WHEN** 用户指定interaction及一个seed运行根脚本
- **THEN** 只运行该条件的既定训练预算，记录结构化协议和A参考身份，不自动追加消融、testing或同步环境
