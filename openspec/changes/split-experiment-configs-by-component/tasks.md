## 1. 设计组件化目录与入口骨架

- [x] 1.1 设计并创建按组件类型划分的配置目录骨架（至少包含 `trainer/`、`model/`、`data_loading/`、`logger/`、`callbacks/`）
- [x] 1.2 约定 experiment-specific 组件文件命名与 defaults 导入方式，并记录到设计/说明中

## 2. 迁移 official experiments

- [x] 2.1 选择一个 item 类 experiment，迁移到“experiment 薄入口 + per-component configs”结构并验证可行性
- [x] 2.2 选择一个 sequence 类 experiment，迁移到相同结构并验证可行性
- [x] 2.3 批量迁移其余 official experiments，确保 trainer/model/data_loading/logger/callbacks 主要参数下沉到对应组件文件

## 3. 验证与收尾

- [x] 3.1 执行 compose / instantiate smoke check，确认 Python 侧仍能通过 `cfg.components` 装配 official experiments
- [x] 3.2 全文检查，确认 experiment 主文件已明显收缩，不再保留大面积字段转发式组件参数映射
