"""共享历史与目录商品残差；cold 行保持零。"""

import copy
import math

import torch
from torch import nn

from src.recommendation.copmrec.mixture import JointMixtureCoPMRec


class SharedCatalogResidual(JointMixtureCoPMRec):
    copmrec_version = "v2"

    def __init__(
        self,
        *,
        residual_scale=1.0,
        residual_lr_multiplier=10.0,
        pretrained_v0_checkpoint=None,
        final_ranking_mode="content",
        ranking_chunk_size=64,
        **kwargs,
    ):
        if final_ranking_mode != "content" or pretrained_v0_checkpoint is not None:
            raise ValueError("Unknown CoPMRec v2 final ranking mode.")
        if isinstance(ranking_chunk_size, bool) or not isinstance(ranking_chunk_size, int) or ranking_chunk_size < 1:
            raise ValueError("Ranking chunk size must be a positive integer.")
        for name, value, allow_zero in (
            ("Residual scale", residual_scale, True),
            ("Residual learning-rate multiplier", residual_lr_multiplier, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value < 0 if allow_zero else value <= 0)
            ):
                raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")
        if (
            kwargs.get("content_only", False)
            or kwargs.get("mechanism_control", "learned_mass") != "learned_mass"
            or kwargs.get("inference_mixture_alpha") is not None
            or kwargs.get("fixed_mixture_alpha") is not None
        ):
            raise ValueError("CoPMRec v2 requires learned alpha and the learned mass protocol.")
        kwargs.setdefault("evaluation_mode", "hybrid")
        kwargs.setdefault("prediction_mode", "hybrid")
        if kwargs["evaluation_mode"] not in {"hybrid", "dense"} or kwargs["prediction_mode"] not in {"hybrid", "dense"}:
            raise ValueError("CoPMRec v2 supports hybrid deployment and dense diagnosis.")
        super().__init__(**kwargs)
        self.residual_scale = float(residual_scale)
        self.residual_lr_multiplier = float(residual_lr_multiplier)
        self.final_ranking_mode = final_ranking_mode
        self.ranking_chunk_size = ranking_chunk_size
        # 直接使用零权重，不消耗 RNG；scale0 对照也不向优化器添加此参数。
        self.collaborative_residual = nn.Embedding.from_pretrained(
            torch.zeros(len(self.item_keys), self.transformer.config.d_model),
            freeze=self.residual_scale == 0,
        )
        self.pretraining_metadata = None
        self._restored_training_contract = None

    def item_content_residual(self, rows):
        if self.residual_scale == 0:
            return None
        return self.collaborative_residual(rows).masked_fill(~self.seen_mask[rows, None], 0) * self.residual_scale

    @property
    def residual_contract(self):
        return dict(
            protocol="copmrec-shared-collaborative-residual-v1",
            residual_scale=self.residual_scale,
            embedding_dim=self.transformer.config.d_model,
            catalog_size=len(self.item_keys),
            sharing="history_and_catalog_after_content_projection",
            cold_policy="seen_mask_zero",
        )

    def configure_optimizers(self):
        if self.training_config.optimizer is None:
            raise ValueError("Optimizer is required for training.")
        residual_parameter = self.collaborative_residual.weight
        parameters = [p for p in self.parameters() if p is not residual_parameter and p.requires_grad]
        optimizer = self.training_config.optimizer(params=parameters)
        if residual_parameter.requires_grad:
            optimizer.add_param_group(
                {
                    "params": [residual_parameter],
                    "lr": optimizer.defaults["lr"] * self.residual_lr_multiplier,
                }
            )
        result = {"optimizer": optimizer}
        if self.training_config.scheduler is not None:
            result["lr_scheduler"] = {
                "scheduler": self.training_config.scheduler(optimizer=optimizer),
                "interval": "step",
            }
        return result

    def on_fit_start(self):
        if (
            self._restored_training_contract is not None
            and self._restored_training_contract["residual_lr_multiplier"] != self.residual_lr_multiplier
        ):
            raise ValueError("CoPMRec v2 resumed training learning-rate multiplier mismatch.")

    def on_predict_start(self):
        if self.trainer.world_size != 1:
            raise ValueError("CoPMRec v2 prediction requires a single GPU/process.")

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["copmrec_collaborative_residual"] = self.residual_contract
        checkpoint["copmrec_collaborative_residual_training"] = dict(residual_lr_multiplier=self.residual_lr_multiplier)
        checkpoint["copmrec_pretraining"] = copy.deepcopy(self.pretraining_metadata)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        residual_contract = checkpoint.get("copmrec_collaborative_residual")
        if residual_contract != self.residual_contract or isinstance(residual_contract.get("residual_scale"), bool):
            raise ValueError("CoPMRec v2 collaborative residual checkpoint contract mismatch.")
        training_contract = checkpoint.get("copmrec_collaborative_residual_training")
        multiplier = training_contract.get("residual_lr_multiplier") if isinstance(training_contract, dict) else None
        if (
            not isinstance(multiplier, (int, float))
            or isinstance(multiplier, bool)
            or not math.isfinite(multiplier)
            or multiplier <= 0
            or set(training_contract) != {"residual_lr_multiplier"}
        ):
            raise ValueError("CoPMRec v2 saved training contract mismatch.")
        if "copmrec_pretraining" not in checkpoint:
            raise ValueError("CoPMRec v2 pretraining metadata mismatch.")
        metadata = checkpoint["copmrec_pretraining"]
        if metadata is not None:
            raise ValueError("CoPMRec requires scratch initialization without pretraining metadata.")
        if self.pretraining_metadata is not None and self.pretraining_metadata != metadata:
            raise ValueError("CoPMRec v2 pretraining metadata mismatch.")
        self._restored_training_contract = copy.deepcopy(training_contract)
        self.pretraining_metadata = copy.deepcopy(metadata)
