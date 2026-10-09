## 1. 授权后的固定规格

- [x] 1.1 依据v5.2结题和用户新增授权，固定共享query辅助CE假设、解释边界与结果对应决定。
- [x] 1.2 固定新增1train／50k＋1singleGPU完整Val、Test0／扫描0，保留旧3train／150k与3Val成本。

## 2. 实现与准备

- [x] 2.1 新v5.3类／exact8辅助契约、21keys CP与核心聚焦验证：architecture实跑新41＋v5.2旧33共74项通过，Ruff通过，旧402运行字节不变；实际生产shape CPU预检与严格checkpoint恢复通过。
- [x] 2.2 新薄model／train-inference experiment、根脚本及完整compose／writer／Bash参数聚焦验证：40项通过，Ruff／Bash语法／独立配置review通过，旧402运行字节逐项保持。
- [x] 2.3 root登记真实新增预算／源码与只读driver，官方同步、实际CPU及DDP2一步smoke；不预记正式完成。

## 3. 唯一正式验证与结题

- [x] 3.1 实际连续50k／100raw点／ownbest／完整状态边界／输入／source审计：hqw189d2终态exit0，主审计与独立复核通过；own-best45000、完整保存状态45000，实际50000预算由完整历史与正常终止另行证明。
- [x] 3.2 ownbest单卡完整Val的175文件／22363用户原始输出、同policy paired双8与v5.2增量审计：6u6g62hk正常exit0；主审计、附加独立复核及三份固定Top10输出的四组只读分析通过，没有新评分或预测。
- [x] 3.3 依真实证据保留／收缩／停止：native R10+7.0572%／N10+13.8794%保留partial证据，parent两增量CI跨0，停止固定辅助CE并保留v5.2主方案；真实登记累计4train／200k、4完整Val、attempt5／失败1、0新Test，单seed与未完成复现保持。
- [x] 3.4 独立代码／证据及结题helper复核通过，root真实终态登记完成，OpenSpec strict通过；效果仅限真实单seed Validation，不声称复现或Testing完成。
