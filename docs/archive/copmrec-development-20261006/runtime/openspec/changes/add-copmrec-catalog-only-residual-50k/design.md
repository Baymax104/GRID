## Context

`Liger.encode`与`dense_logits`动态调用同一个`item_content_residual`。仅覆盖hook或使用scale0会同时移除catalog residual，不能实现目标干预。v5.2完整目录CE、现有scratch optimizer／schedule及部署协议已验证可用。

## Goals / Non-Goals

目标是实现准确的history=false／catalog=true结构、共同连续50k协议和可审查运行入口。没有新增正式额度，不恢复固定辅助CE／双目录平均，不实现teacher、第二query、扫描或阶段续训。当前代码验证只能证明实现，不能证明推荐收益、因果归因或复现。

## Decisions

1. 新`UnifiedCatalogOnlyResidualCoPMRec(UnifiedFullCatalogCECoPMRec)`，版本v5.4；history hook返回None。新`dense_logits`保留父原小段projection／normalize／matmul并显式调用`CollaborativeResidualCoPMRec.item_content_residual(self, rows)`。
2. 不修改旧模块；参数量、state keys、初始化与随机数调用保持。residual仍在catalog CE和mixture内容路径学习，history／SID直接路径取消；query仍由共享encoder／SID表示和原多目标监督共同学习。
3. `residual_placement` exact3：protocol=`copmrec-catalog-only-residual-v1`、history_residual=false、catalog_residual=true。记录在unified scratch契约和writer metadata；checkpoint严格比较字段集合、类型和值。两处sharing表达`catalog_after_content_projection`，不得继续声明history+catalog。
4. 无目录residual辅助CE不进入此类。保留full-catalog CE支持集、seen训练目标要求和cold residual0，三loss各1及learned alpha。
5. 薄配置／根脚本继承v5.2的运行配方，严格替换model及元数据身份；训练DDP2、单卡推理。实际source字节与所有上游输入在正式阶段再核。
6. proposal形成及本地模型实现时新额度尚未批准，未登记reserved／started／committed。2026-10-06T11:01:32.778Z用户明确“批准运行 v5.4”，实际消息及同thread上下文已归档；按原固定范围只登记新增1train50k＋1Val，累计5／250k／5，Test、43、扫描均0。注册、预检与正式启动仍须各自真实凭据；批准本身不是smoke／训练成功。
7. 旧v5.3编排／审计绑定已消费的第4次授权和辅助CE契约；不能因CP同为21键而复用。新本地模板须绑定实际v5.4问题call及真实新增答复，原4／200k／4闭合账本、414源码、exact3 placement和两处sharing；授权缺失时在同步／SSH／job写入前失败。纯fixture不写正式注册、preflight或job成功凭据。v5.2参考保留自身20键与history+catalog sharing，不从v5.4删一字段推定。

## Risks / Trade-offs

移除history residual可能损失协同条件表征。query仍可学习交互信息，并非冻结native query；指标差异不能单独因果分解。更新预算一致不等FLOPs或搜索成本；单seed结果不是整体可复现目标。旧CP即使state keys相同仍属于不同计算语义，必须拒绝。

## Validation

本地内存测试检查零残差初始化一致、非零history残差不改变query而catalog仍改变logits、SID-only residual梯度路径消失／CE-mixture目录梯度有效／cold零、新旧CP语义拒绝以及旧父字节保持。配置compose、writer及实际Bash参数捕获核notes/dry-run/错误输入/额外override，禁止真实Trainer或统一入口的训练链路进入单测。

正式阶段仅在新增授权后执行CPU空optimizer、DDP2一步smoke和一臂随机初始化连续50k、own raw N10首个最佳、单卡175文件／22363用户完整Val；沿用native双8及pairedCI门禁，v5.2只描述增量，无新parent硬门槛。不明确增量即停止此固定结构，不自动追加下一臂。
