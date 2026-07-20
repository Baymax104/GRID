# step-based-training-progress-bar Specification

## Purpose
TBD - created by archiving change adopt-step-based-training-progress-bar. Update Purpose after archive.
## Requirements
### Requirement: 训练主进度条 SHALL 以 step 为主轴显示
对于采用 step-driven 训练预算的训练实验，控制台训练主进度条必须以当前 step 与总 step 预算为核心语义，而非继续突出 epoch 语义。

#### Scenario: 训练进度条显示 step 预算
- **WHEN** 用户运行任一训练实验并观察 train progress bar
- **THEN** progress bar MUST 让用户能够直接判断“当前第几 step / 总 step 预算”
- **THEN** progress bar MUST NOT 以 `Epoch 0/-2` 这类内部 epoch 哨兵值作为主显示语义

#### Scenario: 保留 Rich 风格进度条样式
- **WHEN** 训练主进度条被改为 step-based
- **THEN** 控制台 MUST 使用 Rich 风格的进度条样式（条形、速度、耗时等）
- **THEN** 不得退化成仅输出纯文本的 `Step xx/yy` 日志

#### Scenario: 去掉默认版本号显示
- **WHEN** 用户观察训练主进度条
- **THEN** progress bar MUST NOT 显示 `v_num`

### Requirement: step-based 训练进度条 SHALL 统一适用于所有训练实验
项目中的训练实验不得各自采用互相冲突的 progress bar 主语义。

#### Scenario: 默认训练 callbacks 统一接入
- **WHEN** 用户运行 `rkmeans_train`、`tiger_train`、`rqvae_train`、`rvq_train` 等训练实验
- **THEN** 这些训练实验 MUST 默认使用统一的 step-based training progress bar

