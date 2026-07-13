"""推理结果后处理函数，由 LocalPickleWriter 的 post_processing_functions 配置调用。"""

import torch

from src.utils.tensor_utils import load_model_output


def deduplicate_rows_in_tensor(file_path: str):
    """
    Identifies and de-duplicate repeated rows in a PyTorch tensor.
    Rows that are not duplicated will have a new column with value 0,
    while rows that are duplicated will have a new column indicating the number of duplicates from 1 to N-1
    where N is the number of duplicates for that row.

    Args:
        file_path: Optional; Path to a file containing the tensor data.
    """
    if not file_path.endswith(".pt"):
        return
    model_output = load_model_output(file_path)
    data = model_output.predictions
    assert len(data.size()) == 2, "Input data must be a 2D PyTorch tensor."

    # Use torch.unique to get unique rows and their inverse indices
    unique_rows, inverse_indices, counts = torch.unique(data, dim=0, return_inverse=True, return_counts=True)

    output_indices = torch.zeros_like(inverse_indices)

    # Find indices where counts > 1 (meaning duplicates exist)
    duplicate_indices = torch.where(counts > 1)[0]

    for i in range(len(duplicate_indices)):
        # Calculate number of collisions
        num_of_collisions = counts[duplicate_indices[i]]

        # Gather the indices where the collisions occur
        indices_to_change = torch.where(inverse_indices == duplicate_indices[i])[0]

        # Create a range based on the number of collision, starting from 1
        range_to_add = torch.arange(1, num_of_collisions + 1)

        # Scatter to those specific indices
        output_indices = output_indices.scatter(0, indices_to_change, range_to_add)

    # Concatenate the duplicate indicator column to the original data
    result = torch.cat((data, output_indices.unsqueeze(1)), dim=1).long()
    torch.save({"keys": model_output.keys, "predictions": result}, file_path)
