## Context

当前 `src/utils/__init__.py` 通过 `_EXPORTS` 表和 `__getattr__` 做 lazy export，向外暴露 `RankedLogger`、`extras`、`instantiate_callbacks` 等符号。与此同时，`src/utils` 包内部多个模块仍通过 `from src.utils import pylogger`、`from src.utils import rich_utils` 等方式走 package root 间接引用兄弟模块。用户已明确要求本次变更采用“方案 B”：鼓励直接从子模块导入，并允许顺手修正相关导入以避免循环导入。

## Goals / Non-Goals

**Goals:**
- 移除 `src/utils/__init__.py` 中的动态导出机制。
- 让仓库内主要调用方直接从 `src.utils.<submodule>` 导入需要的符号。
- 清理 `src/utils` 包内部的循环敏感导入路径，减少通过 package root 间接跳转的情况。
- 将 `src/utils/__init__.py` 收缩为极薄入口，避免继续承担聚合 API 职责。

**Non-Goals:**
- 不改变 `utils` 中具体函数或类的业务行为。
- 不重命名现有子模块文件。
- 不对仓库里所有 `src.utils.*` 导入风格做无关统一，只处理本次变更触达的聚合入口与循环风险点。

## Decisions

### 1. 去掉动态 `__getattr__` 导出
- 决策：删除 `src/utils/__init__.py` 中的 `_EXPORTS`、`__getattr__` 与 `import_module` lazy export 逻辑。
- 原因：动态导出会隐藏真实依赖路径，增加阅读、静态分析和循环导入判断成本。
- 备选方案：保留动态导出，仅新增注释说明；被否决，因为不能解决结构不透明问题。

### 2. 调用方改为直接子模块导入
- 决策：把当前 `from src.utils import ...` 的仓库内使用点改成直接从具体子模块导入，例如 `from src.utils.pylogger import RankedLogger`。
- 原因：让依赖关系一眼可见，也减少未来继续依赖 package root 聚合入口的惯性。
- 备选方案：在 `__init__.py` 中改为静态 re-export；被否决，因为用户已明确希望采用方案 B，而不是继续鼓励 package-level 聚合导入。

### 3. utils 包内部避免通过 package root 引兄弟模块
- 决策：对 `utils.py`、`logging_utils.py`、`instantiators.py`、`rich_utils.py` 等内部模块，优先改为直接导入兄弟子模块，而不是 `from src.utils import ...`。
- 原因：这类写法最容易在 package 初始化过程中引入循环敏感路径。
- 备选方案：只改外部调用方；被否决，因为内部导入链才是本次循环导入风险的主要来源。

## Risks / Trade-offs

- [调用方仍残留对聚合入口的依赖] → 通过全文搜索确认 `from src.utils import ...` 的使用点已清理到目标范围内。
- [移除聚合导出后潜在破坏少量隐式依赖] → 通过最小导入 smoke check 和主入口 compose/import 验证控制风险。
- [内部模块互相直接导入后暴露新的循环] → 优先按最短导入路径重排，并对 `src.main`、`launcher_utils`、`instantiators` 等关键链路做导入验证。

## Migration Plan

1. 先梳理 `src/utils` 包内外对聚合导出的依赖点。
2. 将外部调用方改为直接从子模块导入。
3. 再清理 `src/utils` 包内部通过 package root 间接导入兄弟模块的写法。
4. 最后收缩 `src/utils/__init__.py`，并执行导入/compose smoke check。

## Open Questions

- 当前无阻塞性开放问题。
