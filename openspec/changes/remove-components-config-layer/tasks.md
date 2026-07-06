## 1. 调整顶层配置入口

- [x] 1.1 修改 `src/utils/launcher_utils.py`，将 Python 侧读取路径从 `cfg.components.*` 收敛到顶层组件入口
- [x] 1.2 复核与配置路径直接相关的辅助代码/注释，清理对 `components` 容器层的旧假设

## 2. 收敛按组件拆分的配置文件

- [x] 2.1 将 `configs/model/*.yaml`、`configs/trainer/*.yaml`、`configs/data_loading/*.yaml`、`configs/logger/*.yaml`、`configs/callbacks/*.yaml` 改为直接定义顶层组件子树
- [x] 2.2 移除各组件文件中的独立参数域视图与 `components.<group>` 冗余包装
- [x] 2.3 更新 `configs/experiment/*.yaml`，确保 defaults 显式指定挂载位置，并在导入后形成新的顶层组件树

## 3. 验证与收尾

- [x] 3.1 执行 compose / instantiate smoke check，确认 official experiments 可通过新的顶层入口装配
- [x] 3.2 全文搜索确认 `cfg.components` 与 `components.<group>` 不再是官方配置结构依赖
