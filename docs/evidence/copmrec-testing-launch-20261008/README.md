# BMX-116 六个正式Testing启动回执

2026-10-08 六个seed42/200正式Testing全部finished、退出码0，运行中0；用户授权的六条test命令已执行完毕。训练完成6/9、Testing执行完成6/9；输出与用户集合、指标独立复算待核验，完整配对暂不计Done。seed2026未启动。回执：GRID docs/evidence/copmrec-testing-launch-20261008/。

2026-10-08按用户授权启动六个单卡Testing，使用各自审计best v0文件/SHA256，统一group、testing split和独立tmux。6个任务均已验证checkpoint加载、local GPU映射和runtime源码快照。当前W&B running 1 / finished 5，完整单元有效性与独立复算待完成；seed2026未启动。

| Issue | Dataset | Seed | GPU | tmux | Testing run | State |
| -- | -- | -- | -- | -- | -- | -- |
| BMX-120 | beauty | 42 | 7 | copmrec_test_beauty_s42_20261008 | vosmuihm | finished |
| BMX-119 | sports | 42 | 1 | copmrec_test_sports_s42_20261008 | tl55l87o | finished |
| BMX-17 | toys | 42 | 3 | copmrec_test_toys_s42_20261008 | pxh4ffi6 | finished |
| BMX-121 | beauty | 200 | 5 | copmrec_test_beauty_s200_20261008 | 95rx50ma | finished |
| BMX-15 | sports | 200 | 6 | copmrec_test_sports_s200_20261008 | 5dx507jd | running |
| BMX-18 | toys | 200 | 0 | copmrec_test_toys_s200_20261008 | jlod7okq | finished |

实际命令与回执见launch-specs.json、launch-receipt.json和runtime.json。仅GPU编号和独立日志路径按资源调整；未新增训练或Validation，未停止已有GPU进程。
