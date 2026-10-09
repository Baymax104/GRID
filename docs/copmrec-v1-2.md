# CoPMRec v1.2已撤回

2026-10-04，用户要求删除v1.2并退回v1.1。v1.2的模型组件、模型配置、训练/推理experiment与根脚本、专属测试已删除；共享代码中的覆写接口及v1.2 trace支持已撤销。

当前使用 [CoPMRec v1.1](copmrec-v1-1.md)：四路head输入`[h_u,v_i,h_u*v_i,d_ui]`，批量候选评分，排序权重`0.05726763550972437`，推荐模型全部可训练。v0/v1也继续保留。

原实现与规格仅保存在 [历史归档](archive/copmrec-v1-2/README.md)，其中启动命令已失效。此前核验JSON及实验重资产保留，不将撤回当作推荐效果否证。本次未启动或停止任何训练，未删除W&B run、日志或checkpoint，也未新增实验预算。

撤回验证：本地68项模型/loss/恢复/配置/脚本回归通过，2项Linux双进程检查在Windows跳过；Ruff check/format与OpenSpec strict通过。Mutagen flush成功，三个会话Watching、无conflict；node1确认6个运行文件已删除、活动代码无v1.2引用、10个v1.1运行文件指纹匹配。训练和推理脚本以uv替身捕获双卡参数并compose配置，未执行真实训练/推理。结果见 [rollback-verification.json](evidence/copmrec-v1-2-20261004/rollback-verification.json)。
