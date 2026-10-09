## 1. 审查与归档

- [x] 1.1 核对 v5.3 实际继承链、配置、训练和推理契约。
- [x] 1.2 保存历史入口、独有实现、测试和旧 OpenSpec change 原始字节与 SHA256。
- [x] 1.3 移除历史活动入口，保留必要内部基类和 LIGER 公共工具。

## 2. 正式入口

- [x] 2.1 新增统一正式模型和 strict formal checkpoint 契约。
- [x] 2.2 新增正式 model/experiment/trainer，固定 v5.3 数学和 50k 训练配置。
- [x] 2.3 新增双卡训练、单卡 Validation/Testing 脚本及正式标签。
- [x] 2.4 放宽 seed 输入范围，保持已有 seed 的数值行为。
- [x] 2.5 保留 LIGER hybrid 正式 baseline，通过已有排序 hook 统一历史资格且不扩大原候选预算。

## 3. 验证

- [x] 3.1 通过核心模型、正式 checkpoint 和历史资格 CPU 测试。
- [x] 3.2 通过 Hydra compose、Bash quoting/dry-run/notes/override 和 LIGER 保留入口验证。
- [x] 3.3 通过归档摘要校验、静态检查和 OpenSpec strict 验证。
- [x] 3.4 通过 LIGER hybrid 原生成、候选并集、评分、资格和异常清理测试，并 compose 正式 hybrid 命令。
