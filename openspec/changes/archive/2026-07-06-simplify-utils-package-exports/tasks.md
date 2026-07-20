## 1. 梳理与收缩 utils 包入口

- [x] 1.1 盘点仓库内对 `src.utils` 聚合导出的使用点，并确认需要保留或移除的 package root 行为
- [x] 1.2 重写 `src/utils/__init__.py`，移除动态 `__getattr__` / `import_module` 导出机制

## 2. 调整导入路径

- [x] 2.1 将仓库内依赖 `from src.utils import ...` 的调用方改为直接从具体子模块导入
- [x] 2.2 清理 `src/utils` 包内部通过 package root 间接导入兄弟模块的写法，降低循环导入风险

## 3. 验证与收尾

- [x] 3.1 执行最小导入 smoke check，验证关键模块可正常导入
- [x] 3.2 全文搜索确认目标范围内不再依赖 `src.utils` 聚合导出与动态 root export 机制
