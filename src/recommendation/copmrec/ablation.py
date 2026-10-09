"""v2 单组件训练消融；独立 scratch 与独立恢复身份。"""

import torch

from src.recommendation.copmrec.module import CoPMRec

M3_PROTOCOL = "copmrec-v2-ablation-20261009-v1"
VARIANTS = ("no_mixture", "no_residual", "no_joint_ce")


class CoPMRecAblation(CoPMRec):
    def __init__(self, *, variant, **kwargs):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown v2 ablation: {variant!r}.")
        self.variant = variant
        super().__init__(**kwargs)
        if variant == "no_mixture":
            self.dynamic_gate.bias.requires_grad_(False)
        if variant == "no_residual":
            self.collaborative_residual.weight.requires_grad_(False)

    @property
    def ablation_contract(self):
        return dict(
            protocol=M3_PROTOCOL,
            base_release="copmrec-v2",
            variant=self.variant,
            loss_weights=dict(
                sid_ce=1,
                joint_catalog_ce=int(self.variant != "no_joint_ce"),
                mixture_nll=int(self.variant != "no_mixture"),
            ),
            alpha_policy="frozen_0.5" if self.variant == "no_mixture" else "learned",
            residual_mode="frozen_zero_both" if self.variant == "no_residual" else "shared_history_catalog",
        )

    @property
    def formal_release(self):
        return {
            **super().formal_release,
            "evidence_phase": "ablation",
            "variant": self.variant,
            "m3_protocol": M3_PROTOCOL,
        }

    @property
    def unified_scratch_contract(self):
        contract = {
            **super().unified_scratch_contract,
            "m3_ablation": self.ablation_contract,
            "loss_weights": self.ablation_contract["loss_weights"],
            "alpha_policy": self.ablation_contract["alpha_policy"],
            "residual_sharing": self.ablation_contract["residual_mode"],
        }
        if self.variant == "no_residual":
            contract["optimizer"] = {**contract["optimizer"], "residual_lr": None}
        return contract

    def _optimizer_parameters(self):
        if self.variant == "no_residual":
            return [[p for p in self.parameters() if p.requires_grad]]
        return super()._optimizer_parameters()

    def _optimizer_peak_lrs(self):
        return [self.base_lr] if self.variant == "no_residual" else super()._optimizer_peak_lrs()

    def item_content_residual(self, rows):
        residual = super().item_content_residual(rows)
        return torch.zeros_like(residual) if self.variant == "no_residual" else residual

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get("state_dict", {})
        zero_parameter = {"no_residual": "collaborative_residual.weight", "no_mixture": "dynamic_gate.bias"}.get(
            self.variant
        )
        if zero_parameter and torch.count_nonzero(state.get(zero_parameter, torch.ones(1))):
            raise ValueError("Disabled v2 component requires its frozen zero state.")
        super().on_load_checkpoint(checkpoint)

    def _joint_losses(self, *args, **kwargs):
        values = super()._joint_losses(*args, **kwargs)
        total = values["sid_loss"]
        if self.variant != "no_joint_ce":
            total = total + values["content_loss"]
        if self.variant != "no_mixture":
            total = total + values["mixture_loss"]
        else:
            values["mixture_loss"] = values["mixture_loss"].detach()
        values["loss"] = total
        return values
