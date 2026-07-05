## Why

当前 `SequenceDataModule` 与 `ItemDataModule` 共享大部分 stage 管理、文件分配、dataset 实例化与 dataloader 组装逻辑，但两者的继承关系并不反映真实语义。`ItemDataModule` 只是借用 `SequenceDataModule` 复用实现，并绕开了其序列专属 collate 约定，这使 datamodule 结构难以理解，也让后续演进更容易引入重复或错误抽象。

现在需要把 datamodule 层重构成“公共基类 + 两个兄弟 datamodule + 更清晰的文件布局”，让类型关系与职责边界一致，并减少后续修改时的认知负担。

## What Changes

- 提取 datamodule 公共基类 `BaseFileDataModule`，承载共享的 stage 配置、文件分配、dataset 初始化与 dataloader 模板逻辑。
- 将 `SequenceDataModule` 重构为仅保留序列任务专属行为的子类，如序列 collate 构造与相关配置约束。
- 将 `ItemDataModule` 重构为与 `SequenceDataModule` 并列的子类，不再继承后者，仅保留 item 任务专属行为与约束。
- 调整 `src/data/loading/datamodules/` 文件结构，分别拆分为 `base.py`、`sequence.py`、`item.py`。
- 迁移实验配置中的 datamodule `_target_` 路径到新的模块文件，不保留旧兼容路径。

## Capabilities

### New Capabilities
- `datamodule-structure-alignment`: 统一 datamodule 的类型关系与文件结构，使公共装配逻辑收敛到中性基类，序列与 item datamodule 通过兄弟类型分别承载各自语义。

### Modified Capabilities

## Impact

- 受影响代码：`src/data/loading/datamodules/` 下的 datamodule 实现与导出
- 受影响配置：所有引用 `SequenceDataModule` / `ItemDataModule` 的 `configs/experiment/*.yaml`
- 受影响运行装配：Hydra `_target_` 模块路径解析
- 不引入新依赖，不改变外部实验参数语义，但会改变 datamodule 的代码组织与模块路径
