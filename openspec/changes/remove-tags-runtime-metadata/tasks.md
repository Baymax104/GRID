## 1. 删除 tags 运行时逻辑

- [x] 1.1 更新 `src/utils/utils.py`，移除 `extras()` 中的 `enforce_tags` 分支
- [x] 1.2 更新 `src/utils/rich_utils.py`，删除 `enforce_tags()` 与 `tags.log` 写入逻辑
- [x] 1.3 更新 `src/utils/logging_utils.py` 与 `src/utils/__init__.py`，移除 `tags` hparams 和 `enforce_tags` 导出

## 2. 清理配置接口

- [x] 2.1 更新 `configs/extras/default.yaml`，移除 `enforce_tags`
- [x] 2.2 更新所有 official experiment 配置，删除顶层 `tags` 字段
- [x] 2.3 更新 experiment 内 `extras` 配置，使其不再声明 `enforce_tags`

## 3. 同步文档与规格

- [x] 3.1 更新 `AGENTS.md` 中对 `tags` / `enforce_tags` 的说明
- [x] 3.2 更新所有提到 `tags.log` 或 `tags` 运行时语义的 OpenSpec 文档

## 4. 验证

- [x] 4.1 全文检查确认仓库中不再保留运行时 `tags` / `enforce_tags` 依赖
- [x] 4.2 最小验证相关 Python 与 YAML 文件仍可解析
