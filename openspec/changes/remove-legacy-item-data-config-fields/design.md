## Context

当前 item 链路中，`rkmeans_train` 与 `sem_embeds_inference` 已完成迁移，已经不再依赖这些旧协议特征：

- preprocessing 专用派生字段挂在 `dataset_config` 上
- dataloader config 中的 `should_shuffle_rows`
- dataloader config 中额外承载 preprocessing 装配职责

但共享代码层仍然保留了不少兼容痕迹：

- `ItemDatasetConfig` 中残留 `features_to_consider`、`field_type_map`、`embedding_map`、`num_placeholder_tokens_map` 等旧字段
- `ItemDataloaderConfig` 中残留 `preprocessing_functions`、`should_shuffle_rows`、`oov_token`
- `BaseDataModule._get_shuffle_files()` 仍 fallback 到 dataloader config 的 `should_shuffle_rows`

用户已明确：本次只清理 **item 链路** 的旧协议，不触碰 sequence 侧。

## Goals / Non-Goals

**Goals:**
- 让已迁移 item 链路只保留新协议需要的最小配置字段
- 清除会误导后续迁移的旧兼容层
- 为后续 item 类实验迁移提供更干净的模板

**Non-Goals:**
- 不重构 `SequenceDataloaderConfig`
- 不触达 `tiger_*`、semantic id 或 sequence 相关协议
- 不在本次收缩 collate 层或模型层接口

## Decisions

### D1: 仅清理 item 链路共享配置模型
- **选择**：只修改 `ItemDatasetConfig` / `ItemDataloaderConfig`
- **理由**：用户明确要求只清理 item 链路，避免误伤仍未迁移的 sequence 支线

### D2: `BaseDataModule` 去掉 item 链路对 `should_shuffle_rows` 的 fallback
- **选择**：在 item 已迁移场景下，只依赖 `dataset_config.shuffle_files`
- **理由**：继续保留 fallback 会让旧协议在已迁移链路上长期存活

### D3: 以已迁移链路做回归样板
- **选择**：用 `rkmeans_train` 和 `sem_embeds_inference` 验证清理后 contract 是否仍可 compose / instantiate / row-chain 正常
- **理由**：这两条链路覆盖了 item+embedding 与 item+text 两种场景，足以验证 item 侧 contract

## Risks / Trade-offs

- **[风险] 某些尚未迁移的 item 实验可能仍隐式依赖这些旧字段**  
  **缓解**：本次直接接受“旧实验可能无法运行”，以已迁移链路为准。

- **[风险] `BaseDataModule` 去掉 fallback 后，未迁移 item 配置会更早暴露问题**  
  **缓解**：这是有意为之，目的是防止旧协议继续拖延收口。

## Migration Plan

1. 识别 `ItemDatasetConfig` / `ItemDataloaderConfig` 中已迁移 item 链路不再使用的字段
2. 从共享配置模型中删除这些字段
3. 更新 `BaseDataModule` 对 item 新 shuffle contract 的读取逻辑
4. 对 `rkmeans_train` 与 `sem_embeds_inference` 做最小验证
