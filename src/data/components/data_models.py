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
class SIDViews:
    item_ids: torch.Tensor
    raw_sid: torch.Tensor
    model_sid: torch.Tensor
    dedup_digit: torch.Tensor


@dataclass
class RecommendationOutcomeInput:
    """Key-aligned testing labels and generated SID candidates for diagnosis."""

    user_ids: torch.Tensor
    label_item_ids: torch.Tensor
    generated_sids: torch.Tensor


@dataclass
class DiagnosisBatch:
    sid_views: SIDViews
    frequencies: dict[int, int]
    groups_by_item: dict[int, str]
    embeddings: torch.Tensor | None
    recommendation: RecommendationOutcomeInput | None = None
    input_metadata: dict[str, str | None] = field(default_factory=dict)


class ModelOutput:
    """
    推理结果写入的字段规范层，直接持有 keys + predictions。

    Attributes:
        keys: 每条预测对应的业务主键（如 item_id、user_id）。
        predictions: 模型预测值（如 embedding、cluster_ids、semantic_ids）。
    """

    def __init__(self, keys: torch.Tensor, predictions: torch.Tensor):
        self.keys = keys  # (n,)
        self.predictions = predictions  # (n, *)
