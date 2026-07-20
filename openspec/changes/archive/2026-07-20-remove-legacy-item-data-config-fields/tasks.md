## 1. 清理 ItemDatasetConfig 旧字段

- [x] 1.1 删除已迁移 item 链路不再使用的 `ItemDatasetConfig` 旧协议字段
- [x] 1.2 同步更新该 dataclass 的说明文档，使其只描述新 item contract

## 2. 清理 ItemDataloaderConfig 旧字段

- [x] 2.1 删除已迁移 item 链路不再使用的 `ItemDataloaderConfig` 旧协议字段
- [x] 2.2 同步更新该 dataclass 的说明文档，使其只描述新 item contract

## 3. 清理 item 链路旧兼容逻辑

- [x] 3.1 更新 `BaseDataModule`，去掉 item 已迁移链路对 `should_shuffle_rows` 的 fallback
- [x] 3.2 确保 item 链路只通过 `dataset_config.shuffle_files` 读取文件级 shuffle 语义

## 4. 验证收尾

- [x] 4.1 compose / instantiate smoke check：`rkmeans_train` 与 `sem_embeds_inference` 仍可正常解析
- [x] 4.2 row-chain smoke check：两条已迁移 item 链路的 preprocessing 仍工作正常
- [x] 4.3 总结本次清理后 item 新协议的最小字段集合
