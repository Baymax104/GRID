## 1. 实验模块路径对齐

- [x] 1.1 替换代码中残留的 `src.models.embedding|quantization|recommendation.*` 引用为新的顶层 `src.embedding|quantization|recommendation.*`
- [x] 1.2 替换实验配置中残留的旧实验模块 `_target_` 路径
- [x] 1.3 验证默认 train/inference 与实验配置不再依赖旧实验模块路径

## 2. common 目录重命名

- [x] 2.1 将 `src/models/common` 迁移为 `src/common`
- [x] 2.2 替换代码中 `src.models.common.*` 引用为 `src.common.*`
- [x] 2.3 替换配置中 `src.models.common.*` 的 `_target_` 路径

## 3. 验证

- [x] 3.1 做全文搜索，确认旧路径引用已清理完成
- [x] 3.2 做最小静态检查，确认迁移后导入与配置路径可解析
- [x] 3.3 复核 `src/common` 中的模块边界符合“整体公有模块”约定
