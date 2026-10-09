## Context
作者 LETTER-TIGER 使用随机初始化 T5，保留32100基础token，按字典序追加目录出现的code token。
## Goals / Non-Goals
独立骨干与合法生成，数据装配及日志生命周期留给后续提案。
## Decisions
- 独立 LetterBackbone 直接装配第三方 T5ForConditionalGeneration，保留共享 embedding、EOS/pad及标准shift。
- 官方 tokenizer 的拼接code编码已本地核验：新token从32100开始，结尾EOS=1；整数映射按相同字符串字典序建立，不下载语言权重。
- CE只在训练logits除tau；generate不缩放，不在合法mask后重归一化，明确length_penalty=1。
- SID目录必须四层、唯一且整数合法；Trie包含start=0与EOS=1。候选最终转为原始商品keys。
## Risks / Trade-offs
新旧Transformers版本不能声明逐位相同；用标准库和小目录合法性/温度公式测试保证协议。
