"""只使用内存小模型，不运行 Trainer 或真实数据。"""

import copy
from functools import partial
from unittest.mock import patch

import torch
from test_liger import batch, model

from src.common.configs.model import TrainingModelConfig
from src.common.scheduler import WarmupCosineSchedulerNonzeroMin
from src.recommendation.copmrec import CoPMRec
from src.recommendation.copmrec.ablation import CoPMRecAblation


def copmrec(variant=None, **kwargs):
    config = TrainingModelConfig(
        optimizer=partial(torch.optim.AdamW, lr=0.0003, weight_decay=0.035),
        scheduler=partial(WarmupCosineSchedulerNonzeroMin, warmup_steps=2500, scheduler_steps=50000, min_ratio=0.0),
    )
    options = dict(training_model_config=config)
    options.update(kwargs)
    if variant is not None:
        options["variant"] = variant
    with patch("test_liger.Liger", CoPMRec if variant is None else CoPMRecAblation):
        return model(**options)


def snapshot(variant=None, updates=1, **kwargs):
    torch.manual_seed(42)
    actual = copmrec(variant, **kwargs)
    result = actual.configure_optimizers()
    optimizer, scheduler = result["optimizer"], result["lr_scheduler"]["scheduler"]
    for _ in range(updates):
        optimizer.zero_grad()
        actual.losses(batch()[0], batch()[1].target_ids, training=True)["loss"].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actual.parameters() if p.requires_grad)
        optimizer.step()
        scheduler.step()
    saved = dict(
        global_step=updates,
        state_dict=copy.deepcopy(actual.state_dict()),
        optimizer_states=[copy.deepcopy(optimizer.state_dict())],
        lr_schedulers=[copy.deepcopy(scheduler.state_dict())],
    )
    actual.on_save_checkpoint(saved)
    return actual, saved
