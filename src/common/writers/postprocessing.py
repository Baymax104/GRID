"""Post-processing functions for prediction output files."""

import torch

from src.data.components.artifacts import load_model_output


def deduplicate_rows_in_tensor(file_path: str):
    model_output = load_model_output(file_path)
    keys = model_output.keys
    predictions = model_output.predictions

    if predictions.ndim != 2:
        raise ValueError(
            "Deduplication expects 2-D prediction tensor with shape "
            f"(num_rows, num_hierarchies), got {tuple(predictions.shape)}."
        )

    dedup_digit = torch.zeros(predictions.size(0), dtype=predictions.dtype)
    seen: dict[tuple[int, ...], int] = {}
    for idx, row in enumerate(predictions.tolist()):
        row_key = tuple(int(value) for value in row)
        dedup_digit[idx] = seen.get(row_key, 0)
        seen[row_key] = seen.get(row_key, 0) + 1

    updated_predictions = torch.cat([predictions, dedup_digit.unsqueeze(1)], dim=1)
    torch.save({"keys": keys.cpu(), "predictions": updated_predictions.cpu()}, file_path)
