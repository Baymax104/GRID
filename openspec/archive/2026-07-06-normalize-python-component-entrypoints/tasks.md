## 1. 调整 Python 侧实例化入口

- [x] 1.1 修改 `src/utils/launcher_utils.py`，让 datamodule 和 model 的顶层实例化直接读取 `components` 下的装配根
- [x] 1.2 修改 callback/logger/trainer 的 Python 侧读取路径，统一改为 `components.callbacks`、`components.logger`、`components.trainer.root`
- [x] 1.3 复核日志/辅助代码对配置路径的直接访问，清理仅服务于旧入口的路径假设

## 2. 重组 official experiment 的组件入口命名

- [x] 2.1 为 item 类 experiment 统一引入并对齐 `components.data_loading.datamodule`、`components.model.root`、`components.trainer.root`、`components.callbacks`、`components.logger`
- [x] 2.2 为 sequence 类 experiment 统一引入并对齐上述组件入口命名
- [x] 2.3 删除参数域中仅用于 Python 直接实例化的中转别名，确保 `data_loading`、`model` 等域主要保留参数和值引用

## 3. 验证与收尾

- [x] 3.1 做配置 compose / instantiate smoke check，确认 official experiments 能通过新的 Python->components 路径装配
- [x] 3.2 做全文搜索，确认旧的 Python 实例化入口路径不再作为官方配置依赖
- [x] 3.3 视需要更新说明性注释，明确新的入口规范为 `python -> components config -> argument config`
