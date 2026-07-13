import torch


def delete_module(module: torch.nn.Module, module_name: str):
    """Recursively delete a submodule from a module.

    :param module: the parent module that we want the submodule to be removed from.
    :param module_name: the name of the submodule to be removed.
    :return: None.
    """
    if hasattr(module, module_name):
        delattr(module, module_name)

    for _name, submodule in module.named_children():
        delete_module(submodule, module_name)


def find_module_shape(module: torch.nn.Module, module_name: str) -> torch.Size | None:
    """Recursively find a submodule in a module and return its shape.

    :param module: the parent module that we want the submodule to be removed from.
    :param module_name: the name of the submodule to be removed.
    :return: the shape of the module if it exists.
    """
    if hasattr(module, module_name):
        return getattr(module, module_name).weight.shape

    for _name, submodule in module.named_children():
        shape = find_module_shape(submodule, module_name)
        if shape:
            return shape
    return None


def reset_parameters(module: torch.nn.Module):
    """Reset the parameters of a given module.

    :param module: the module whose parameters will be reset.
    :return: None.
    """

    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    else:
        for layer in module.children():
            reset_parameters(layer)


def get_parent_module_and_attr(model: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    """
    Get the parent module and attribute name for a given module name.

    Args:
        model (torch.nn.Module): The model containing the module.
        module_name (str): The full name of the module.

    Returns:
        tuple[torch.nn.Module, str]: The parent module and the attribute name.
    """
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def create_last_k_mask(sequence_length: int, last_item_index: torch.Tensor, last_k: int | None = None) -> torch.Tensor:
    """
    Creates a mask to select the last K items of sequences.
    If a sequence has less than K items, all items are considered for the row.
    If last_k is None, all items are considered for all rows.

    Args:
        sequence_length (int): The length of the sequences.
        last_item_index (torch.Tensor) of shape (batch_size,).
            The tensor containing the indices of the last items in the each row
        last_k (int | None): The number of last K items to consider.
            If None, all items are considered.
    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, sequence_length) with
            True for the last K items in each row and False for the rest.
    """

    if last_k is None:
        start_index = torch.zeros_like(last_item_index)
    else:
        if last_k < 1:
            raise ValueError("last_k must be None or greater than or equal to 1")
        start_index = torch.clamp(last_item_index - last_k + 1, min=0)  # Shape (batch_size,)

    indices = (
        torch.arange(sequence_length, device=last_item_index.device).unsqueeze(0).expand(last_item_index.size(0), -1)
    )  # shape (batch_size, sequence_length)

    # Shape (batch_size, sequence_length)
    mask = (indices >= start_index.unsqueeze(1)) & (indices <= last_item_index.unsqueeze(1))
    return mask
