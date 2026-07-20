## 1. 清理未使用 default 配置

- [x] 1.1 确认 `trainer/default.yaml`、`logger/default.yaml`、`callbacks/default.yaml` 是否仍被 official 入口引用
- [x] 1.2 删除已确认无引用的 default 配置，并评估/清理连带失去引用的 callback 模板文件

## 2. 将 data_loading 更名为 data

- [x] 2.1 将 `configs/data_loading/` 重命名/迁移为 `configs/data/`，并更新 official experiment defaults 挂载路径
- [x] 2.2 批量更新 split config 中的自引用与跨组件引用：`data_loading.*` → `data.*`
- [x] 2.3 修改 Python 侧读取路径、日志字段、warning 与 config tree 打印中的 `data_loading` 命名

## 3. 验证与收尾

- [x] 3.1 执行 compose / instantiate / import smoke check，确认 official 入口链路正常
- [x] 3.2 全文搜索确认 `data_loading` 与被删除的 default 配置在官方链路中不再残留
