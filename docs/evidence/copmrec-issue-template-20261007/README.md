# CoPMRec issue 模板与评价流程统一回执

2026-10-07按用户要求，将9个CoPMRec主结果issue统一为其他实验issue使用的六段结构：实验单元、当前状态、固定协议与输入、从仓库根目录启动（训练／Testing）、当前正式结果、Done门禁。

- 标题统一为“[主结果/CoPMRec] 数据集 / seed”，实现版本v5.3仅在描述中说明。
- 每个单元只保留原训练命令与原Testing命令；参数和quoting不变。
- 删除额外独立Validation命令、结果登记栏、Testing前置要求及Done要求。训练内每500步validation和best选点保留，best审计后直接Testing。
- 父BMX-116同步结构与预算；BMX-117及3个机制子任务同步取消待准备消融的额外Validation，不改变未实现/未运行状态。
- 主实验新增预算为9训练／450k更新／0额外Validation／9Testing；内部dense另列9Testing，消融仍待准备。
- 研究状态、当前计划、正式版本、实验计划、Run Registry与Linear文档同步。validation_run=null标记not_required，不作为待执行缺口。

原LIGER9组正式结果、原issue内容与Done/有效保留。本次未修改模型、训练配置或启动任何实验；训练内validation不会因模板修改关闭。

修改前快照见issues-before.json；提交内容见issue-proposals.json；在线回读及本地文档/命令检查见本目录验证文件。Linear会自动规范Markdown转义与issue提及，正文以规范化比较、命令以逐字比较核对。
