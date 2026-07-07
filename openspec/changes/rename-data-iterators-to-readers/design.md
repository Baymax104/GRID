## Context

当前原始数据读取组件位于 `src/data/components/iterators.py`，并通过 `data_iterator` 字段注入到 `SequenceDatasetConfig` / `ItemDatasetConfig`。但从职责看，这些对象并非只负责 `__iter__` 风格的最小迭代协议，而是提供：

- 文件路径集合与后缀知识
- 文件级与行级读取策略
- 可配置的 shuffle 行为
- 面向 dataset / datamodule 的原始数据读取抽象

因此“reader”比“iterator”更能表达其在项目中的角色。用户已经明确两点：

1. **整套一起改** —— 不仅改字段名，也接受模块名、类名、配置名、文档统一调整；
2. **接受修改 spec** —— 可以推翻此前 `iterators.py` 保留不动的设计决策。

## Goals / Non-Goals

**Goals:**
- 统一项目内原始数据源组件的命名语义为 reader
- 避免出现字段叫 `data_reader`、类型却仍叫 `BaseIterator` 的半重构状态
- 让代码、配置、文档、spec 对同一概念使用一致术语
- 将本次变更明确记录为对历史命名决策的修订

**Non-Goals:**
- 不改变 reader 的读取逻辑、shuffle 策略、文件切分方式
- 不重构 dataset / datamodule 的控制流
- 不引入兼容别名层长期并存；目标是统一迁移到 reader 语义

## Decisions

### D1: 采用整套 reader 命名收敛，而非只改字段名
- **选择**：字段名、类名、模块名、Hydra `_target_`、文档术语一起改。
- **理由**：用户已明确要求“整套一起改”；只改字段名会形成 `data_reader: BaseIterator` 这类命名不一致状态。
- **备选**：仅把 `data_iterator` 改成 `data_reader` —— 否决，语义混杂。

### D2: `iterators.py` 直接改为 `readers.py`
- **选择**：模块文件名同步使用 reader 语义。
- **理由**：若类名与字段名都改为 reader，模块名继续保留 `iterators.py` 会残留旧概念。
- **备选**：只改类名和字段名，保留模块名 —— 否决，一致性不足。

### D3: 配置 contract 统一使用 `data_reader`
- **选择**：`SequenceDatasetConfig`、`ItemDatasetConfig` 及 YAML 顶层 key 统一改为 `data_reader`。
- **理由**：配置是维护者最常接触的入口，术语需要与代码语义一致。

### D4: 不保留长期兼容壳
- **选择**：不为 `iterators.py` / `data_iterator` 保留长期兼容别名。
- **理由**：本次是内部命名收敛，仓库内可一次性全量迁移；保留兼容壳会延长双命名状态。
- **备选**：增加过渡别名 —— 否决，收益低且增加维护成本。

### D5: 显式修订既有 spec 决策
- **选择**：在本次 change 中修改 `data-loading-package-layout` 对 `iterators.py` 的要求，并新增 reader contract 规格。
- **理由**：历史 proposal/design/spec 已明确写出“`iterators.py` 保留不动”，本次必须把这次反转记录为显式 spec 变化。

## Risks / Trade-offs

- **[风险] Hydra 配置引用面广，容易漏改某个 `data_iterator` / `_target_`**  
  **缓解**：实现时对 `configs/data/*.yaml`、`src/` imports、文档进行全量 grep 校验。

- **[风险] 文档和 spec 的旧术语残留，会让 reader / iterator 并存**  
  **缓解**：把文档与 OpenSpec 一并纳入本次改动范围，而不是只改代码。

- **[权衡] 本次属于语义命名优化，不直接产生功能收益**  
  **可接受**：其价值在于降低认知成本、统一领域术语，并修正历史命名决策。

## Migration Plan

1. 重命名 reader 模块与 reader 类
2. 更新配置 dataclass、dataset、datamodule 对应字段与引用
3. 更新全部 Hydra YAML key 与 `_target_`
4. 更新文档与注释术语
5. 更新 OpenSpec 中 `iterators.py` 的相关规格与任务记录
6. 通过 grep / import / Hydra compose 做最小验证，确保无旧命名残留
