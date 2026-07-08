## Context

`rkmeans_train` 已完成以下收敛：

- reader factory contract
- `shuffle_files` / `shuffle_rows` contract
- row-only preprocessing
- 配置文件显式声明 preprocessing chain

而 `sem_embeds_inference` 当前仍保留旧模式：

- 顶层 `dataset` 节点
- `preprocessing.*` 分散定义后再聚合
- 通过 resolver 派生 `features_to_consider` / `field_type_map` 等中间字段
- dataloader 中保留 `should_shuffle_rows`

该实验的优点是：

- 仅有 predict data 链路，没有 train / val / test 的多分支复杂度
- preprocessing 逻辑清晰，包含文本 tokenization，可补足 `rkmeans_train` 没覆盖到的 text 路径

因此适合作为第二个迁移模板。

## Goals / Non-Goals

**Goals:**
- 将 `sem_embeds_inference` 迁到新的 data config / preprocessing contract
- 保持 preprocessing 参数显式可读
- 统一 reader factory 和 shuffle contract
- 尽量减少 resolver 中间派生层

**Non-Goals:**
- 不要求本次一定删除 `features`
- 不改变 `sem_embeds_inference` 的模型行为与输出语义
- 不同时迁移其他实验

## Decisions

### D1: 顶层 dataset config 节点改名为 `predict_dataset_config`
- **选择**：用更明确的命名表达其仅服务 predict dataloader
- **理由**：与 `rkmeans_train` 的 `train_dataset_config` / `eval_dataset_config` 风格保持一致，也避免继续沿用含糊的 `dataset`

### D2: preprocessing chain 改为顶层公共块显式声明
- **选择**：像 `rkmeans_train` 一样，在配置顶层定义 `preprocessing_functions`，然后注入 `predict_dataset_config`
- **理由**：配置一眼可读，避免 `preprocessing.*` 分散后再聚合

### D3: 去掉仅服务 preprocessing 的 resolver 派生字段
- **选择**：将 `features_to_consider`、`field_type_map` 等参数直接写进 preprocessing config，而非先经 resolver 派生到 dataset config
- **理由**：与当前“可读性优先”的方向一致

### D4: reader 与 shuffle contract 对齐新模板
- **选择**：
  - `data_reader` 使用 `_partial_` factory 形式
  - dataset config 显式提供 `shuffle_files`
  - reader config 显式提供 `shuffle_rows`
  - predict dataloader 不再以 `should_shuffle_rows` 作为目标 contract
- **理由**：统一到已在 `rkmeans_train` 验证过的模板

### D5: `features` 暂不强制删除
- **选择**：本次不把删除 `features` 作为完成条件，待迁移完成后再判断其剩余用途
- **理由**：当前优先级是快速平稳迁移，而不是过早追求最小配置面

## Risks / Trade-offs

- **[风险] 文本预处理链比 `rkmeans_train` 多一步 tokenizer，参数显式直写后配置可能更长**  
  **缓解**：可接受，用户当前优先级是显式可读。

- **[风险] 若 `features` 仍被别处隐式引用，强删会引入额外回归**  
  **缓解**：本次先不把删除 `features` 作为硬目标。

## Migration Plan

1. 将 `dataset` 改名为 `predict_dataset_config`
2. 把 preprocessing chain 收敛为顶层 `preprocessing_functions`
3. 直接在 preprocessing 配置中显式写参数，去掉 resolver 中间派生字段
4. 统一 `data_reader` factory + `shuffle_files` / `shuffle_rows`
5. 做最小 compose / instantiate / row-chain 验证
