## 1. 固定问题与预算

- [x] 1.1 核当前source、保留正反证据与用户新目标，确定直接scratch及同raw checkpoint选择／同history部署。
- [x] 1.2 登记single50k日程、双8%门槛、seed43成对复现、stage3train/150k/3Val/3Test及旧额度封存。

## 2. 实现与预检

- [x] 2.1 实现统一scratch薄模型、严格来源／50k／own full-state checkpoint契约及核心聚焦测试。
- [x] 2.2 新增薄Hydra配置和根训练／推理脚本，核compose、notes/dry-run/quoting/override及shell语法。
- [x] 2.3 独立交叉review、actualruntime字节与官方Mutagen flush、真实CPU初始化及双卡一步smoke通过。

## 3. 首个完整scratch证据

- [x] 3.1 唯一seed42随机初始化连续50k正式训练，审计实际终止预算/own raw Valbest/已保存optimizer日程/source及上游；实际best/last均41k，没有50k完整状态文件，预算由100次Val及max_steps终止日志单独证明。
- [x] 3.2 ownbest完整单卡eligible Validation与原50k native42同口径独立重算，判定双8及pairedCI，不用Testing选择；8w893ra3实际R10+3.60%且CI跨零、N10+11.29%且CI为正，双8门槛未通过，保留NDCG正向结果并分析已保存bad case。

## 4. 冻结方案复现与验收

- [ ] 4.1 首个Val晋级后冻结方法、native43与samecandidate43各完整50k并审计完整状态与same-runtime公平性。
- [ ] 4.2 两个43 ownbest完整单卡Val，冻结所有最终引用后最多三次新单卡Testing，复用native42固定输出。
- [ ] 4.3 每seed独立raw用户/labels/keys/合法性/输入/CP/source/pairedmetrics复核，记录双8是否可复现及全部成本边界。
- [ ] 4.4 实际结果交叉review、report/research-state/current-plan同步和OpenSpec严格验证；全目标未满足保持active。
