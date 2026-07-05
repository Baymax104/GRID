## 1. 提取公共 datamodule 基类

- [x] 1.1 新增 `src/data/loading/datamodules/base.py` 并定义 `BaseFileDataModule`
- [x] 1.2 将 stage 配置管理、文件分配、dataset 初始化与公共 dataloader 组装逻辑迁移到基类
- [x] 1.3 在基类中定义子类差异 hook（如配置校验与 collate 构造），避免两个具体 datamodule 复制 `get_dataloader()`

## 2. 拆分 sequence 与 item datamodule 文件

- [x] 2.1 新增 `src/data/loading/datamodules/sequence.py`，仅保留 `SequenceDataModule` 的序列专属行为
- [x] 2.2 新增 `src/data/loading/datamodules/item.py`，仅保留 `ItemDataModule` 的 item 专属行为与约束
- [x] 2.3 更新 `src/data/loading/datamodules/__init__.py` 导出新模块结构，并移除旧的跨语义继承关系

## 3. 迁移配置与验证

- [x] 3.1 更新 `configs/experiment/*.yaml` 中 datamodule `_target_` 到 `src.data.loading.datamodules.sequence.SequenceDataModule` 或 `src.data.loading.datamodules.item.ItemDataModule`
- [x] 3.2 做全文搜索，确认仓库内不再依赖旧的 combined datamodule 模块路径
- [x] 3.3 做最小 smoke check，确认 datamodule 导入与 Hydra 配置路径仍可解析
