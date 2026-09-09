## 1. W&B 配置 taxonomy

- [x] 1.1 将 Tiger train/inference 和 Tail-SID diagnosis experiment 的 `group` 改为必填输入，并保留量化与 semantic embedding experiment 的固定 group。
- [x] 1.2 将全部官方 W&B logger run name 更新为 `${task_name}/${now_tz:%Y-%m-%d_%H-%M-%S}`，保持现有 job type 和本地输出 identity。

## 2. 官方启动脚本

- [x] 2.1 为 `tiger_train.sh` 和 `tiger_inference.sh` 增加必填 `--group` 两种语法、允许值校验和 Hydra 透传。
- [x] 2.2 为 `tail_sid_diagnosis.sh` 增加同样的 group 参数契约，并保持 notes、Artifact 路径、dry-run 与尾部 override 行为。

## 3. 测试与验证

- [x] 3.1 更新 Hydra compose/W&B identity 测试，覆盖固定及动态 group、必填失败、三种量化方法和 run name 格式。
- [x] 3.2 扩展启动脚本测试，覆盖两种 group 语法、缺失/非法值、RQ-VAE、quoted options、尾部 override 和 Bash 语法。
- [x] 3.3 运行聚焦 pytest、Hydra compose 检查和所有受影响脚本的 `bash -n`。
- [x] 3.4 运行 `openspec validate group-wandb-runs-by-quantization-method --strict`。
