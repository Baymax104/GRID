## 1. 删除配置与类型定义

- [x] 1.1 从数据加载相关 config/dataclass 定义中删除 `assign_all_files_per_worker` 字段
- [x] 1.2 从当前量化训练 experiment 配置中移除 `assign_all_files_per_worker`

## 2. 删除运行时逻辑

- [x] 2.1 删除 datamodule 中围绕 `assign_all_files_per_worker` 的校验与参数透传逻辑
- [x] 2.2 删除 dataset / worker 文件选择中围绕“全部 worker 共享全部文件”的分支逻辑
- [x] 2.3 删除文件分配工具函数中围绕该能力的输入参数和实现分支

## 3. 清理与验证

- [x] 3.1 清理相关注释与文档，避免继续描述已删除的 worker 策略
- [x] 3.2 做全文搜索，确认官方代码与配置中不再残留 `assign_all_files_per_worker`
- [x] 3.3 做最小静态检查，确认数据加载相关 Python 模块仍可导入解析
