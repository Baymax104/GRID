## 1. 恢复配置直写 preprocessing chain

- [x] 1.1 在 `configs/data/rkmeans_train.yaml` 中显式声明 `train_dataset_config` / `eval_dataset_config` 的 `preprocessing_functions`
- [x] 1.2 每个 preprocessing 参数直接写为字面量或局部可见加载配置，不再依赖 resolver 派生

## 2. 删除 assembler 中间层

- [x] 2.1 更新 `SequenceDataset`，直接消费 `dataset_config.preprocessing_functions`
- [x] 2.2 删除 `src/data/components/preprocessing_assembly.py`
- [x] 2.3 清理 `config_models.py` 中仅服务 assembler 的字段（如 `preprocessing_steps`、`features`，若无其他用途）

## 3. 保持纯函数接口不回退

- [x] 3.1 确保 preprocessing 函数继续不接收 `dataset_config`
- [x] 3.2 确保配置中传入的仍是最小必要参数，而不是新的宽泛 config 容器

## 4. 验证收尾

- [x] 4.1 grep 验证 `rkmeans_train` 主链路不再依赖 `preprocessing_assembly` / `preprocessing_steps` / `features`
- [x] 4.2 import / compose / instantiate smoke check：`rkmeans_train` 的 preprocessing 配置可被 dataset 正常消费
- [x] 4.3 row-chain 验证：单条 row 经过配置声明的 preprocessing chain 后仍得到预期结果
