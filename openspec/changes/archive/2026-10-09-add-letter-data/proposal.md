## Why
LETTER需要保持作者逐前缀监督与EOS，现有模型专用preprocessor不能作为依赖。
## What Changes
- 新增独立目录、keyed内容/CF输入、逐前缀推荐数据与collate。
- 数据容器与公共reader/bundle协议复用；不包含模型训练或脚本。
## Capabilities
### New Capabilities
- `letter-data`: 共同split与商品key下的LETTER输入契约。
### Modified Capabilities
无。
## Impact
新增src/data/components/letter.py；公共artifact字段新增CF角色。
