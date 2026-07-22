from dataclasses import dataclass, field

import torch


@dataclass
class TigerLabelData:
    """TIGER training/evaluation labels."""

    target_ids: torch.Tensor


@dataclass
class TigerModelInput:
    """TIGER model input batch.

    Attributes:
        input_ids: Flattened semantic ID sequence tensor for the encoder.
        attention_mask: Encoder attention mask derived from ``input_ids``.
        output_keys: Optional prediction keys used only when writing inference outputs.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    output_keys: torch.Tensor | None = None


@dataclass
class ItemBatch:
    """The data class used to wrap a batch of item features.

    Parameters
    ----------
    item_ids: torch.Tensor
        The item ids.
    features: dict[str, torch.Tensor]
        The transformed features.
    """

    item_ids: torch.Tensor
    features: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ItemTextBatch(ItemBatch):
    """The data class used to wrap a batch of items with text features for training/testing.

    It is a child class of ItemData, with additional text tokens and text masks.

    Parameters
    ----------
    text_tokens: torch.Tensor | None
        The text tokens.
    text_masks: torch.Tensor | None
        The text masks.
    """

    text_tokens: torch.Tensor | None = None
    text_masks: torch.Tensor | None = None
