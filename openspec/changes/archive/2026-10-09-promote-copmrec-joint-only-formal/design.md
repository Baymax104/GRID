## Context

原 v0/v1 使用四项训练损失；A3 独立训练删除 native CE，v1 推理结果已核验。用户现在选择这一设计为新正式方法；版本冻结不授权完整运行。

## Goals / Non-Goals

Goals：独立正式入口仅计算三项训练损失；商品残差继续受 joint CE 和其余目标监督；原始历史输入、完整目录、cold 零残差、稳定排名及50k/双卡日程保持；推理无历史排除；文档与 Linear 一致。

Non-Goals：不追改历史身份、不重新选点、不启动主矩阵或新消融、不证明共享参数双目标必然错误、不将用户区间解释为训练seed重复验证。

## Decisions

- 新类 `CoPMRecJointOnly` 复用现存 `UnifiedFullCatalogCECoPMRec` 三损失实现，独立 formal release / checkpoint 契约。替代方案仅将 native 权重置零仍计算无用评分，不满足删除 native view；直接改旧类会破坏已完成来源。
- 新模型配置继承 LIGER 组件并明确匹配原 CoPMRec 参数，加入 mixture 日志与双卡同步指标；避免继承旧 native metric 后出现不存在的 payload。
- 根脚本复用原正式脚本的参数解析，后置新 experiment；额外用户 override 继续优先。推理 SHA 仍为必须输入，使用原统一入口/Artifact恢复逻辑。
- 方法编号与 Artifact 版本分离；用户已选择 v2，保存原 v1 定义。A3保留消融来源，记录为本次正式设计的选型证据，不写成新正式矩阵完成。
- 研究与 Linear 保留旧记录并加最新权威定义；不创建没有检查点的伪可运行 Testing issue，不撤回已完成任务。

## Risks / Trade-offs

- [不同接口下随机数/梯度漂移] → 对比A3初始状态、dropout随机数、三损失和全部参数梯度。
- [配置仍有native metric或旧排除metadata] → compose实际脚本并实例化MetricEngine，核验新来源契约与布尔值。
- [新版本冒充A3或v0完整验证] → 保留原run/artifact，声明选型/覆盖范围、0运行预算。

## Migration Plan

新增独立入口、完成CPU聚焦验证、记录定义/边界和Linear最新状态。正式运行由用户另行启动；旧入口继续用于旧版本，回退无需改历史文件。

## Open Questions

编号已由用户确认为v2，无未决实现问题。
