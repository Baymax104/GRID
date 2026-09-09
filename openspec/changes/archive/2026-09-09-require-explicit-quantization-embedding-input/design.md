## Context

六个官方量化脚本目前都在 `ARGS` 中写死 `embedding_path=wandb://vb8es5ow`，尽管对应 experiment 已在顶层声明 `embedding_path: ???`，data component 和 Artifact resolver 也已经支持本地路径及 `wandb://` URI。脚本因此成为唯一残留的固定数据源绑定，并绕过了配置层已经建立的人工输入契约。

这些脚本已有一致的 Bash 参数解析结构：操作参数被转换为默认 Hydra overrides，未知参数保存在 `EXTRA_ARGS` 并最后追加。训练脚本还要求 notes；推理脚本还要求 checkpoint。此次设计只扩展这套入口契约。

## Goals / Non-Goals

**Goals:**

- 强制六个量化脚本显式选择 embedding 数据源。
- 同时支持本地路径、`wandb://` 引用和两种常用 CLI 参数形式。
- 保持带空格路径作为单个 Hydra 参数传递。
- 保留现有 dry-run、notes、checkpoint、设备参数和尾部 override 优先级。
- 用聚焦的脚本测试锁定六个入口的一致行为。

**Non-Goals:**

- 不改变 `embedding_path` 的 Hydra 字段名或 experiment 必填状态。
- 不在 Bash 中解析、下载或验证 W&B Artifact。
- 不通过环境变量或 producer metadata 隐式推断 embedding。
- 不修改 RKMeans、RVQ、RQVAE 模型、数据加载或 Artifact lineage。

## Decisions

1. 六个脚本统一增加必填 `--embedding-path`。

   每个脚本初始化 `EMBEDDING_PATH=""`，解析 `--embedding-path=*` 与 `--embedding-path <value>`。相比保留历史默认值，必填参数能让每次运行的数据来源在命令和 W&B config 中都可审计。

2. Bash 只验证“已提供非空值”。

   separated syntax 在没有后续 token 或后续 token 以 `--` 开头时，以状态码 2 失败；解析结束后再次校验非空。脚本不限制必须是 `wandb://`，因为现有数据层明确支持本地路径，也不重复 Artifact resolver 的语义验证。

3. 将 embedding override 放入默认参数，尾部 Hydra override 保持最后追加。

   脚本构造 `embedding_path="$EMBEDDING_PATH"`，Bash 数组保证空格不被拆分。`EXTRA_ARGS` 继续最后追加，因此项目既有的高级用户尾部覆盖契约不变。仅提供原始 `embedding_path=...` 而不使用正式 flag 仍会失败；这样可保持脚本接口明确，避免扫描和解释任意 Hydra override。

4. 共享参数化测试覆盖全部六个脚本。

   测试通过替换 `uv` 的轻量 Bash harness 捕获最终 argv，不运行真实 experiment。用参数化用例覆盖 train/inference 差异、两种 flag 形式、本地/W&B 路径、缺失值、dry-run 和尾部 override；另以 `bash -n` 验证语法，并静态禁止历史固定 URI。

## Risks / Trade-offs

- [Risk] 现有自动化未传 `--embedding-path` 会立即失败。→ 这是有意的 breaking change；错误文本给出本地路径和 `wandb://` 两种有效输入，并在迁移时更新调用命令。
- [Risk] 尾部 `embedding_path=...` 可覆盖正式 flag。→ 保留项目既有的 Hydra override 优先级，测试明确该顺序；W&B resolved config 仍记录最终值。
- [Risk] 六份脚本存在重复解析代码。→ 本次沿用现有简单脚本结构，避免为一个参数引入共享 shell library；参数化测试负责防止行为漂移。
- [Risk] Windows 风格路径或含空格路径可能被拆分。→ 始终使用 Bash 数组和双引号保存、转发参数，并增加带空格路径测试。

## Migration Plan

1. 为六个脚本增加参数解析、校验和动态 Hydra override。
2. 更新调用方，在 notes/checkpoint 之外显式增加 `--embedding-path`。
3. 运行脚本契约测试、`bash -n`、相关 pytest 和 OpenSpec strict validation。
4. 如需回滚，可恢复固定 URI 和移除 flag；不涉及数据或 checkpoint 格式迁移。

## Open Questions

无。
