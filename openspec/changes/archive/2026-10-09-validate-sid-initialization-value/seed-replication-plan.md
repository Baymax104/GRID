# Beauty seed43/44固定实现复核

## 固定范围

用户授权仅新增四次训练：A43、残差43、A44、残差44。复用seed42结果，不重跑first_only，不新增模块、不扫超参。

A固定为`token_content_init` + `full_content`，使用原catalog组件及相同初始化数学运算；残差固定为`deep_residual` + `matched-code-geometry-v1`。核心文件与配置、依赖清单的交付SHA256见`frozen-implementation.json`；这是当前文件快照，不等同于新Git提交或运行时锁定。后续分析需核对resolved config和产物身份。

仅改变训练seed43/44；bank_seed固定42、PCA固定内容目录、projection_dim128、SID/内容/量化器来源沿用已核查版本。每run从头训练、20k steps、双卡每卡batch128、lr0.0005、每500step验证，best-val NDCG@10规则不变。测试在42/43/44均验证相同seed下backbone/随机流及A原实现逐位兼容。

## 两终端并行启动

从node1仓库根目录`/data3/weizhenyu/projects/GRID`执行。使用近期实验实际占用的两组GPU4/5及6/7；如需其他空闲GPU对，仅修改`--gpus`。

终端一：seed43，A之后依次运行残差。

```bash
bash ./tiger_content_initialization_train.sh \
  --data-dir data/beauty --condition paired --seed 43 \
  --gpus 4,5 --master-port 29770 \
  --notes "Frozen A versus matched deep residual; Beauty seed43 paired replication; unchanged 20k protocol"
```

终端二：seed44，A之后依次运行残差。

```bash
bash ./tiger_content_initialization_train.sh \
  --data-dir data/beauty --condition paired --seed 44 \
  --gpus 6,7 --master-port 29771 \
  --notes "Frozen A versus matched deep residual; Beauty seed44 paired replication; unchanged 20k protocol"
```

合计4次训练、80k steps；任意时刻最多2个训练任务，各占两张卡。同一seed在相同GPU对上顺序完成两个条件，两个seed之间可并行。任一条件失败则该终端停止，不自动恢复或重跑。

如某seed的A已完成，只需残差，使用相同seed和GPU参数，将`--condition paired`改为`--condition deep_residual`，并追加`+initialization_replication=beauty-seed43-44-v1 +initialization_pair_seed=43`（seed44对应44）。单独A使用`--condition full_content`。勿重复运行paired导致已有条件重训。

原`both`仍表示first_only+deep_residual，不用于本轮。paired自动记录`initialization_replication=beauty-seed43-44-v1`及`initialization_pair_seed`。支持dry-run/notes两种语法及末尾Hydra覆盖；复核命令不要额外改预算/模型/seed，否则须重新判断可比性。

## 预先声明的判读

- 收齐四个finished run并核对40次验证、输入digest和配置，仅按同seed配对比较。
- 主指标每seed最佳val NDCG@10，附对应Recall；补充共同19000/20000步与最后5次验证均值，检查选择点是否主导结果。
- 将已有seed42与新增43/44一起列出每seed差值和三seed均值/标准差；不把40个验证点当40次独立实验。
- 若两个新seed均支持A、后段曲线一致，继续固定A并准备独立测试/跨数据集验证；不因单次seed特别好而追加调参。
- 若方向冲突或接近零，记录稳定性不足，先解释现有三seed证据，不自动引入模块或替换假设。
- 三seed仍不足以支持强统计结论；evaluation参与checkpoint选择，不能当独立测试泛化。四次训练完成后不自动启动推理或追加实验。

## 交付状态

45项针对性测试通过，包含seed43/44实际脚本参数的Hydra compose、每seed初始化等价与随机流、已有脚本语法/quoting/错误输入。OpenSpec严格验证通过。Mutagen flush成功且四个session均Watching for changes、无冲突。未修改模型实现，未运行完整训练或创建Git提交。
