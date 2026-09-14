"""冻结 hybrid checkpoint 的训练集 teacher-forcing 熵标定。"""

import torch

from src.data.components.data_models import ModelOutput


@torch.no_grad()
def calibration_output(model, encoded, mask, inputs, labels):
    if model.data_split != "training" or labels is None or not model.loaded_checkpoint_fingerprint:
        raise ValueError("Calibration needs training labels and a loaded checkpoint.")
    targets = labels.target_ids.long()
    model.catalog.item_indices(targets)
    states = model.states(encoded, mask, targets[:, :-1])
    entropy = []
    for depth in range(model.num_hierarchies):
        lp = model.route_log(states[:, depth], targets[:, :depth])
        entropy.append(-(lp.exp() * lp.masked_fill(~lp.isfinite(), 0.0)).sum(-1))
    values = torch.stack(entropy, 1)
    payload = dict(
        schema_version="item_resolution_calibration_v1",
        labels=targets.cpu(),
        trace={"entropy": values.cpu()},
        metadata=dict(
            source_split="training",
            checkpoint_fingerprint=model.loaded_checkpoint_fingerprint,
            checkpoint_reference=model.checkpoint_reference,
            catalog_fingerprint=model.catalog.fingerprint,
            calibration_rule="mean_teacher_forcing_entropy_per_layer",
            num_hierarchies=model.num_hierarchies,
        ),
    )
    return ModelOutput(
        keys=inputs.output_keys.cpu(), predictions=values.cpu(), auxiliary={"item_resolution_calibration": payload}
    )
