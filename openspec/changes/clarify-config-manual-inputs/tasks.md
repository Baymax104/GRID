## 1. 建立可读性基线

- [x] 1.1 整理 `configs/train.yaml` 与 `configs/inference.yaml` 的 section 顺序、块注释与空行风格
- [x] 1.2 整理 `configs/paths/default.yaml`，将 `data_dir` 改为透传顶层变量而非独立手填入口

## 2. 清晰化 experiment 手动输入入口

- [x] 2.1 在代表性 experiment 配置中建立统一的 Manual inputs 与 Runtime metadata 分区
- [x] 2.2 将 experiment 深层裸 `???` 占位改为引用顶层手动输入字段，或移除无意义残留占位
- [x] 2.3 统一较长 experiment 配置的 section 顺序、块注释与空行风格

## 3. 验证

- [x] 3.1 验证整理后 YAML 仍可解析
- [x] 3.2 全文检查确认主要手动输入字段已集中到顶层入口，深层不再保留重复手填入口
