# LETTER 推荐训练停止与 W&B 清理（2026-10-09）

用户授权停止 node1 LETTER 推荐训练，并清除对应弃用的 W&B run 和 Artifact。本次范围仅为 BMX-60～65 的六组未完成推荐训练，不删除已验证的正式上游。

## 已完成操作

- 按六个准确 tmux session、pane PID、运行ID、仓库cwd和当前UID核对进程树。向六个 torchrun launcher 发送 SIGINT；共追踪150个子进程，优雅退出完成，无需强制终止。复查相关 DDP/data worker 和指定端口训练命令均不存在。tmux会话及日志窗口保留。
- 在 W&B 删除六个推荐训练 run：vzzddvhg、oremp4gb、od6uvwdi、h628mv5l、gzi64g6z、xzf4m2f0。
- 删除六个对应源码 Artifact 版本：grid-source-<上述run ID>:v0。每个版本的生产者为目标run，且没有其他run消费；没有删除整个共享collection。
- 这些训练没有发布 checkpoint Artifact。停止后 W&B 曾标为finished，但最后记录step仅11499～19499，不表示完成50k预算；按用户要求弃用并删除。

## 删除后核验

新建W&B API客户端查询确认六个run均不再出现在项目中，六个源码Artifact均返回not_found。六个SID Artifact仍为COMMITTED；内容、CF导出、Tokenizer和SID共20个直接上游run仍可查询，六个CF teacher训练run另行核验仍保留。

节点本地日志、checkpoint、源码归档和速度诊断证据保留。未启动修复后的训练或Testing，未更改模型实现，也未清理其他用户的GPU进程。

## 记录

node1：/data3/weizhenyu/projects/GRID/logs/letter-cancel-20261009/；本地副本：logs/letter-cancel-20261009/。processes-before/after、wandb-deletion-inventory、deletion-result、verification.json分别登记停止范围、停止后结果、精确产物清单、删除回执与远端读回核验。

此前 docs/letter-launch-20261008.md 和速度诊断中的run链接为历史记录，所指六个推荐run现已删除，不能作为当前运行或正式结果引用。有效SID可继续用于后续修复后的独立运行。

