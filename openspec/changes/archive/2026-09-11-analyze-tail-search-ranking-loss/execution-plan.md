# 四设置重分析执行清单

## 可追溯状态

- 基线提交：`9459906d379bec2ad01d3418c5babf861f2a4805`
- 代码状态：包含尚未提交的 `add-tiger-prefix-survival-instrumentation` 前置实现与本提案实现；运行时 W&B notes 必须明确记录 dirty 前置状态。
- 数据划分：`evaluation`
- 随机种子：`42`
- 固定 beam：`10`
- widened beam：`50`
- 主静态风险配置：5 箱，每组每箱至少 20 items，最低 item retention 0.5，`|raw_damage SMD| <= 0.1`
- 敏感性配置：3、5、10 箱；全部结果保留。
- Bootstrap：沿用 diagnosis 主配置，prefix cluster 重采样，固定 seed 42。

## 输入身份

| 设置 | data_dir | group | fixed recommendation/trace | widened recommendation/trace | semantic ID | embedding | checkpoint reference |
|---|---|---|---|---|---|---|---|
| Beauty RKMeans | `data/beauty` | `rkmeans` | `wandb://m4h0geda` | `wandb://56rrarps` | `wandb://4vyi4o6w` | `wandb://3jtt9mpa` | `wandb://26qh50do` |
| Beauty RVQ | `data/beauty` | `rqvae` | `wandb://r94ut5sv` | `wandb://k7mr3xlk` | `wandb://d2hhqdic` | `wandb://3jtt9mpa` | `wandb://ye9u9yj7` |
| Sports RKMeans | `data/sports` | `rkmeans` | `wandb://5024wy48` | `wandb://4x3steyq` | `wandb://3narllqy` | `wandb://psec3u5i` | `wandb://129w8p0r` |
| Sports RVQ | `data/sports` | `rqvae` | `wandb://lzttakbs` | `wandb://n6svfy9g` | `wandb://ykntf4ve` | `wandb://psec3u5i` | `wandb://49ote174` |

每次执行通过 `tail_sid_diagnosis.sh` 传入上述引用，并追加：

```text
calibration_statistics_ready=true
search_ranking.enabled=true
risk_standardization.enabled=true
```

根脚本继续保证 `--data-dir` 必填、seed 默认 42、notes/dry-run 支持以及末尾 Hydra override 最高优先级。
