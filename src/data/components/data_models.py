from dataclasses import dataclass, field

import torch


@dataclass
class LabelFunctionOutput:
    """Class to unify the output of label functions, making it easier to merge those
    into SequentialModelInputData and SequentialModuleLabelData

    Parameters:
    -----------
    sequence: torch.Tensor
        The sequence tensor of shape (batch_size, sequence_length).
    labels: torch.Tensor
        The labels tensor of shape (num_labels,).
    label_location: torch.Tensor
        The label location tensor of shape (num_labels, 2).
        This is used to indicate the position of the labels in `sequence`.

    """

    sequence: torch.Tensor
    labels: torch.Tensor
    label_location: torch.Tensor = None
    attention_mask: torch.Tensor = None


@dataclass
class SequentialModuleLabelData:
    """The label data class used to wrap the label data for training/testing.

    Parameters
    ----------
    labels: dict[str, torch.Tensor]
        Dictionary of label_name to label tensor.
        Label tensor is the shape of mask size # long tensor
    label_location: dict[str, torch.Tensor]
        Dictionary of label_name to label location tensor.
        Label location tensor is the shape of mask_size, 2 as it contains coordinates # long tensor
    """

    labels: dict[str, torch.Tensor] = field(default_factory=dict)
    label_location: dict[str, torch.Tensor] = field(default_factory=dict)
    attention_mask: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class SequentialModelInputData:
    """The model input data class used to wrap the model input data for training/testing.

    Parameters
    ----------
    user_id_list: torch.Tensor | list[str] | None
        Tensor or list of user_ids.
    transformed_sequences: dict[str, torch.Tensor]
        Dictionary of sequence_name to sequence tensor.
        Sequence tensor is (batch_size_per_device x sequence length)
    mask: torch.Tensor
        The mask for the sequence data.
        (batch_size_per_device x sequence length)
    """

    user_id_list: torch.Tensor | list[str] | None = None
    transformed_sequences: dict[str, torch.Tensor] = field(default_factory=dict)
    mask: torch.Tensor = (
        None  # Single mask if needed as all sequences are padded the same way.
    )


@dataclass
class ItemData:
    """The data class used to wrap a batch of item features.

    Parameters
    ----------
    item_ids: torch.Tensor | list[str] | None
        The item ids.
    transformed_features: dict[str, torch.Tensor]
        The transformed features.
    """

    item_ids: torch.Tensor | list[str] | None = None
    transformed_features: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ItemTextData(ItemData):
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
