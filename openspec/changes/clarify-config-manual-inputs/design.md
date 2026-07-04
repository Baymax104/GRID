## Context

当前配置文件的主要问题不在于功能失效，而在于“阅读接口”不清晰：较长 experiment 文件缺少统一视觉分块，顶层 section 顺序不稳定，用户需要手动填写的字段又和运行元信息、深层派生配置交织在一起。尤其是部分深层结构仍保留裸 `???` 占位，例如 `semantic_ids: ???`，而真正的用户入口已经在 experiment 顶层，导致配置链路对人类读者不够直观。

本轮目标是做一轮“清晰化重构”：统一视觉结构，同时把用户需要直接填写的字段集中到 experiment 顶层，深层结构只消费这些顶层字段，不再暴露第二套手填入口。

## Goals / Non-Goals

**Goals:**
- 在主入口与较长 experiment 配置中建立统一的 section 顺序。
- 在较长配置中加入明显视觉分割注释块。
- 将用户手动输入字段集中到 experiment 顶层统一位置。
- 将深层裸 `???` 占位改为引用顶层字段，或移除无意义残留项。
- 将 `paths/default.yaml` 中的 `data_dir` 改为透传顶层变量，而不是再次作为手填入口。

**Non-Goals:**
- 不改变训练/推理实际语义。
- 不回收 experiment 中重复的 `trainer` / `logger` / `paths` 结构。
- 不修改 OpenSpec 历史文档。

## Decisions

### 1. 采用统一的 section 顺序与块注释风格
- 决策：长配置文件采用固定 section 顺序，并使用 `# -----------------------------` 风格的块注释进行视觉分割。
- 原因：这是提升扫读性的最低风险手段。

### 2. 顶层建立明确的 Manual inputs 区域
- 决策：experiment 文件开头集中放置所有用户需要手动填写的字段，例如 `data_dir`、`embedding_path`、`semantic_id_path`、`devices`、`num_hierarchies` 等。
- 原因：让用户一打开文件就知道需要修改哪些字段。

### 3. 深层配置不再保留第二套手填入口
- 决策：像 `semantic_ids: ???`、`embeddings: ???` 这类深层占位要么引用顶层变量，要么删除无意义残留占位。
- 原因：避免“一条链路中到处都是 ???”的认知负担。

### 4. 默认层中的 `data_dir` 改为透传
- 决策：`configs/paths/default.yaml` 中的 `data_dir` 不再作为独立手填入口，而改为透传顶层 `data_dir`。
- 原因：用户不应同时面对 experiment 顶层和默认层两处 `data_dir: ???`。

## Risks / Trade-offs

- [纯重排引入意外语义变化] → 仅做结构与引用整理，保持字段值和插值关系不变，并做 YAML 解析验证。
- [删除深层 `???` 后降低局部可见性] → 通过块注释和顶层 Manual inputs 区域弥补。
- [不同 experiment 的输入字段集合不同] → 允许每个 experiment 的 Manual inputs 区域字段不同，但位置和样式统一。

## Migration Plan

1. 先整理 `train.yaml` / `inference.yaml` / `paths/default.yaml` 作为风格基线。
2. 再整理代表性 experiment（embedding、quantization、recommendation）。
3. 清理深层裸 `???`，改为引用顶层字段。
4. 做 YAML 解析与必要的全文搜索验证。

## Open Questions

- 当前无阻塞性开放问题。
