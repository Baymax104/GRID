## Why

当前配置文件虽然可以运行，但人工阅读成本较高：较长 experiment 文件缺少统一的视觉分隔，顶层 section 顺序不稳定，而“需要用户手动传入”的字段又与运行元信息、深层派生配置混杂在一起。结果是用户很难一眼看出哪些参数必须填写、哪些只是内部消费链路的一部分。

现在需要做一次清晰化重构：在不改变配置语义的前提下，统一配置文件的视觉结构，并把手动输入字段集中到更清晰的顶层入口，消除深层裸 `???` 占位带来的困惑。

## What Changes

- 为主入口与较长 experiment 配置建立统一的 section 顺序、注释风格与空行规则。
- 在较长配置文件中加入明显的视觉分割注释块，例如 `# -----------------------------` 风格标题。
- 将用户需要手动传入的字段集中到 experiment 顶层统一区域。
- 将深层配置中的裸 `???` 占位改为引用顶层手动输入字段，或删除无意义残留占位。
- 调整默认层（如 `paths/default.yaml`）中会与 experiment 顶层重复暴露手动输入入口的字段，使其改为透传而不是再次要求用户填写。

## Capabilities

### New Capabilities
- `config-manual-input-clarity`: 让配置文件具备统一视觉结构，并将用户手动输入字段集中在清晰的顶层入口中。

### Modified Capabilities

## Impact

- 受影响文件预计包括：`configs/train.yaml`、`configs/inference.yaml`、`configs/paths/default.yaml`，以及多个 `configs/experiment/*.yaml`
- 不涉及运行时逻辑实现变更，目标是提升可读性与配置入口清晰度
- 需要小心保持 Hydra 插值关系与现有实验行为不变
