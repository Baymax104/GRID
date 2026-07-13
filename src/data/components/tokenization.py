from functools import partial
from typing import Any


def load_tokenize(config: Any):
    """Load tokenizer and return a partial function for tokenization."""
    tokenizer = config.tokenizer
    tokenize = partial(
        tokenizer,
        max_length=config.max_length,
        padding=config.padding,
        truncation=config.truncation,
        add_special_tokens=config.add_special_tokens,
        return_tensors="pt",
    )
    return tokenize
