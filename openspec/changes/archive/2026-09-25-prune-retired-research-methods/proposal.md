## Why

仓库同时保留了多轮已结题研究方法、诊断入口和当前 CoPMRec/LIGER 主线，导致配置面、启动脚本和模块职责远大于论文复现实务需要。当前方法已经冻结，应收缩到可投稿实验矩阵实际使用的实现，降低误启动旧路线和维护失效依赖的风险。

## What Changes

- **BREAKING**：删除已结题的 BRIR、MIR/item-resolution、CGBS/catalog-grounded、TIGER training probe 方法实现及其专用配置、启动脚本、writer 和测试。
- **BREAKING**：删除已结题的 LIGER 动态门控、学习排序、深度条件聚合、候选并集、偏好分散和来源保护实现及其专用入口。
- 将 CoPMRec 需要的固定概率混合逻辑收拢到 LIGER 核心模块，移除对已淘汰动态门控模块的依赖。
- 保留 TIGER 基线、LIGER 基线、CoPMRec 主方法、legal/max 机制控制、candidate trace、embedding、quantization、统一 launcher 和 W&B lineage。
- 清理已删除模块在 Artifact loader、配置 defaults、测试和根目录脚本中的引用，并增加活跃方法表面回归检查。

## Capabilities

### New Capabilities

- `active-research-method-surface`: 定义代码仓库只暴露当前论文路线和必要基线的运行入口，并禁止已淘汰方法重新出现在可执行配置中。

### Modified Capabilities

- `unused-config-pruning`: 将已结题研究方法的配置与启动入口纳入必须删除的无效配置范围。
- `module-path-alignment`: 收缩 recommendation 模块布局，使每个保留目录对应当前可执行方法或共享基础设施。

## Impact

- 删除 `src/recommendation/`、`src/data/`、`src/common/writers/` 下的历史方法专用模块。
- 删除对应 `configs/`、根目录脚本、`scripts/` helper 和聚焦测试。
- 修改 LIGER 核心实现、Artifact loader、配置与结构检查测试。
- 历史论文记录、W&B run 和 Artifact 证据不随代码删除；旧命令将不再可从当前仓库执行。
