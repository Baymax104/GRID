## Why

`src/utils/__init__.py` 目前通过 `__getattr__` 和 `import_module` 提供动态导出，增加了导入链不透明性，也让 `src.utils` 包内部的循环导入风险更难判断。现在需要把 `utils` 包的入口整理为更明确的静态结构，并推动调用方直接从子模块导入。

## What Changes

- **BREAKING** 收缩 `src/utils/__init__.py`，移除当前基于动态属性解析的聚合导出行为。
- 将仓库内依赖 `from src.utils import ...` 的调用点改为直接从具体子模块导入。
- 清理 `src/utils` 包内部通过 package root 间接引用兄弟模块的写法，改为更直接的子模块导入，以降低循环导入风险。
- 保留 `src.utils` 作为 package 边界，但不再鼓励将其作为主要公共聚合入口。

## Capabilities

### New Capabilities
- `utils-package-static-imports`: `src.utils` 包 SHALL 采用静态、可追踪的导入结构，并允许调用方稳定地从具体子模块导入。

### Modified Capabilities

## Impact

- 受影响代码：`src/utils/__init__.py`、`src/utils/*.py` 内部导入、以及当前使用 `from src.utils import ...` 的调用方
- 受影响 API：仓库内部对 `src.utils` 聚合导出的使用方式
- 不引入新依赖，不改变业务功能行为，主要影响 import structure 和 package surface
