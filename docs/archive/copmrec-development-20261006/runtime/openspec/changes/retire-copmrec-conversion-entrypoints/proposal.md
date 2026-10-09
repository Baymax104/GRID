## Why

CoPMRec 基础版本已有 BMX-116 的有效训练和推理结果，但相对 LIGER dense 的最终推荐收益转化仍未完成。用户要求清理基于冻结模型的旧探索启动入口，保留做法与效果作为参考，再设计全参数联合训练的收益转化组件。

## What Changes

- **BREAKING**：移除 decoder 微调、独立 ranker、冻结 teacher scratch、旧 mixed 终排及候选缓存的 10 个根脚本和 9 个 Hydra experiment 入口。
- 将以上入口和 4 份仅针对这些入口的配置/脚本测试按原始字节存入文档 ZIP 归档，保存路径、大小、SHA256 和归档哈希。
- 用入口退役和基础配置可用性检查替换旧启动契约测试；保留算法实现、组件配置和纯内存算法测试作为技术参考。
- 更新研究定位和旧流程退出记录；收益转化组件的重新设计单独记录，不在本清理变更中实现新模型或启动实验。
- 通过现有 Mutagen 入口传播删除，保护远端数据、日志、checkpoint、环境和 W&B 产物。

## Capabilities

### New Capabilities

- `copmrec-entrypoint-retirement`：旧收益转化实验入口退出启动范围，原始文件可追溯，基础训练/推理和保留算法测试继续可用。

### Modified Capabilities

无 living spec 修改；既有 change 中的实验要求按原日期保留，并由本次退役记录标明其入口已退出。

## Impact

影响根目录旧脚本、`configs/experiment/` 的九份旧配置、四份旧入口测试、新退役检查和 `docs/` 归档。基础 `liger_joint_train/inference`、LIGER/SASRec/LETTER/TIGER 入口与模型实现不变。本地已存在未提交改动，操作严格限定到归档清单，不进行 Git 提交、历史操作或全仓库清理。
