"""正式 scratch 训练与完整优化器恢复协议。"""

import math

import torch
from lightning.pytorch.trainer.states import TrainerFn

from src.common.scheduler import WarmupCosineSchedulerNonzeroMin
from src.recommendation.copmrec.residual import SharedCatalogResidual
from src.recommendation.liger.module import Liger


def _same_number(actual, expected):
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)
    )


class ScratchTrainingProtocol(SharedCatalogResidual):
    copmrec_version = "v2"
    max_updates = 50000
    base_lr = 0.0003
    residual_lr = 0.002
    lr_multiplier = 20 / 3

    def __init__(
        self,
        *,
        initialization_seed=42,
        pretrained_v0_checkpoint=None,
        residual_scale=1.0,
        residual_lr_multiplier=20 / 3,
        final_ranking_mode="content",
        **kwargs,
    ):
        if (
            type(initialization_seed) is not int
            or initialization_seed < 0
            or pretrained_v0_checkpoint is not None
            or not _same_number(residual_scale, 1)
            or not _same_number(residual_lr_multiplier, self.lr_multiplier)
            or final_ranking_mode != "content"
        ):
            raise ValueError(
                "Unified scratch requires a nonnegative integer seed, no pretrained weights and the fixed residual protocol."
            )
        kwargs.setdefault("evaluation_mode", "dense")
        kwargs.setdefault("prediction_mode", "dense")
        if (
            kwargs["evaluation_mode"] != "dense"
            or kwargs["prediction_mode"] != "dense"
            or kwargs.get("candidate_trace", False)
            or kwargs.get("path_trace", False)
        ):
            raise ValueError("Unified scratch requires fixed dense selection/deployment without target traces.")
        super().__init__(residual_scale=1, residual_lr_multiplier=self.lr_multiplier, **kwargs)
        self.initialization_seed = initialization_seed
        self.restored_step = None
        self._fit_started = False
        self._training_state_checked = False

    @property
    def unified_scratch_contract(self):
        return dict(
            protocol="copmrec-unified-scratch-50k-v1",
            version=self.copmrec_version,
            initialization="random_recommender_zero_gate_and_residual",
            initialization_seed=self.initialization_seed,
            pretrained_checkpoint=None,
            max_updates=self.max_updates,
            global_batch_size=256,
            optimizer=dict(name="AdamW", base_lr=self.base_lr, residual_lr=self.residual_lr, weight_decay=0.035),
            scheduler=dict(warmup_steps=2500, scheduler_steps=self.max_updates, min_ratio=0.0, num_cycles=0.5),
            loss_weights=[1, 1, 1],
            alpha_policy="learned",
            residual_scale=1.0,
            residual_sharing="history_and_catalog_after_content_projection",
            cold_residual="seen_mask_zero",
            training_validation="raw_full_catalog_dense",
            checkpoint_selection="val/ndcg@10",
            validation_interval=500,
            deployment="raw_full_catalog_dense_without_history_exclusion",
            rank_ties="stable_catalog_row",
        )

    def eval_step(self, batch):
        model_input, label = batch
        if label is None:
            raise ValueError("Unified scratch checkpoint selection requires labels.")
        losses = self.losses(model_input, label.target_ids)
        # 验证与部署均使用无排除的完整目录 dense 排序。
        sids, scores = Liger.retrieve(self, model_input, "dense")
        return {**losses, "generated_ids": sids, "labels": label.target_ids, "marginal_probs": scores}

    def _optimizer_parameters(self):
        residual = self.collaborative_residual.weight
        return [[p for p in self.parameters() if p is not residual and p.requires_grad], [residual]]

    def _optimizer_peak_lrs(self):
        return [self.base_lr, self.residual_lr]

    def _validate_optimizer_state(self, groups, state, scheduler_state, step, *, saved):
        parameters = self._optimizer_parameters()
        peak_lrs = self._optimizer_peak_lrs()
        expected_schedule = self.unified_scratch_contract["scheduler"]
        if (
            not isinstance(groups, list)
            or len(groups) != len(parameters)
            or not isinstance(state, dict)
            or not isinstance(scheduler_state, dict)
            or any(scheduler_state.get(key) != value for key, value in expected_schedule.items())
            or scheduler_state.get("base_lrs") != peak_lrs
            or scheduler_state.get("last_epoch") != step
            or scheduler_state.get("_step_count") != step + 1
        ):
            raise ValueError(
                "Unified scratch optimizer/scheduler must preserve the single warm2500/cosine50000 schedule."
            )
        factor = (
            step / 2500 if step < 2500 else 0.5 * (1 + math.cos(math.pi * (step - 2500) / (self.max_updates - 2500)))
        )
        last_lrs = scheduler_state.get("_last_lr")
        if (
            not isinstance(last_lrs, list)
            or len(last_lrs) != len(peak_lrs)
            or any(not _same_number(actual, peak * factor) for actual, peak in zip(last_lrs, peak_lrs, strict=True))
        ):
            raise ValueError("Unified scratch scheduler learning-rate state mismatch.")
        ids = []
        for group, expected_parameters, peak in zip(groups, parameters, peak_lrs, strict=True):
            if (
                not isinstance(group, dict)
                or len(group.get("params", [])) != len(expected_parameters)
                or not _same_number(group.get("initial_lr"), peak)
                or not _same_number(group.get("lr"), peak * factor)
                or not _same_number(group.get("weight_decay"), 0.035)
                or group.get("betas") != (0.9, 0.999)
                or not _same_number(group.get("eps"), 1e-8)
                or group.get("amsgrad") is not False
                or group.get("maximize") is not False
            ):
                raise ValueError("Unified scratch requires the fixed AdamW parameter groups and current learning rate.")
            ids.extend(group["params"])
            if not saved and any(
                actual is not expected for actual, expected in zip(group["params"], expected_parameters, strict=True)
            ):
                raise ValueError("Unified scratch optimizer parameter groups mismatch.")
        if len(set(ids)) != len(ids) or set(state) != (set(ids) if step else set()):
            raise ValueError("Unified scratch requires a complete optimizer state, without weights-only resets.")
        if step:
            for identifier, parameter in zip(ids, [p for group in parameters for p in group], strict=True):
                moment = state[identifier]
                moment_step = moment.get("step") if isinstance(moment, dict) else None
                if isinstance(moment_step, torch.Tensor) and moment_step.numel() == 1:
                    moment_step = moment_step.item()
                if (
                    not isinstance(moment, dict)
                    or not _same_number(moment_step, step)
                    or set(moment) != {"step", "exp_avg", "exp_avg_sq"}
                ):
                    raise ValueError("Unified scratch requires every AdamW moment at the restored global step.")
                for name in ("exp_avg", "exp_avg_sq"):
                    value = moment[name]
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.shape != parameter.shape
                        or value.dtype != torch.float32
                        or not torch.isfinite(value).all()
                    ):
                        raise ValueError("Unified scratch optimizer moment shape/dtype/finite contract mismatch.")

    def configure_optimizers(self):
        result = super().configure_optimizers()
        optimizer = result["optimizer"]
        scheduler = result.get("lr_scheduler", {}).get("scheduler")
        if type(optimizer) is not torch.optim.AdamW or not isinstance(scheduler, WarmupCosineSchedulerNonzeroMin):
            raise ValueError("Unified scratch requires AdamW and the fixed warmup cosine scheduler.")
        self._validate_optimizer_state(optimizer.param_groups, optimizer.state, scheduler.state_dict(), 0, saved=False)
        return result

    def on_fit_start(self):
        super().on_fit_start()
        trainer = self.trainer
        smoke = trainer.max_steps == 1 and trainer.limit_train_batches == 1 and trainer.limit_val_batches == 0
        data = trainer.datamodule.get_stage_config(TrainerFn.FITTING)
        if (
            self._fit_started
            or self.pretraining_metadata is not None
            or trainer.world_size != 2
            or trainer.accumulate_grad_batches != 1
            or data.batch_size_per_device != 128
            or data.drop_last is not True
            or not smoke
            and trainer.max_steps != self.max_updates
            or trainer.val_check_interval != 500
            or trainer.gradient_clip_val != 1
            or trainer.precision != "32-true"
            or any(p.dtype != torch.float32 for p in self.parameters())
            or self.restored_step == self.max_updates
            or smoke
            and (
                self.restored_step is not None or any("wandb" in type(logger).__module__ for logger in trainer.loggers)
            )
        ):
            raise ValueError("Unified scratch requires one DDP2/global256, FP32, continuous 0-to-50000 fit.")
        if len(trainer.optimizers) != 1 or len(trainer.lr_scheduler_configs) != 1:
            raise ValueError("Unified scratch requires one optimizer and one continuous scheduler.")
        optimizer, scheduler = trainer.optimizers[0], trainer.lr_scheduler_configs[0].scheduler
        if type(optimizer) is not torch.optim.AdamW or not isinstance(scheduler, WarmupCosineSchedulerNonzeroMin):
            raise ValueError("Unified scratch actual optimizer/scheduler type mismatch.")
        self._fit_started = True

    def on_train_start(self):
        # Lightning 在 on_fit_start 之后恢复 loops/optimizer/scheduler；此处才校验真实进度。
        trainer = self.trainer
        step = trainer.global_step
        if (
            not self._fit_started
            or self._training_state_checked
            or type(step) is not int
            or not 0 <= step < self.max_updates
            or step != (0 if self.restored_step is None else self.restored_step)
            or trainer.max_steps == 1
            and self.restored_step is not None
        ):
            raise ValueError("Unified scratch training progress must preserve its own continuous checkpoint budget.")
        optimizer, scheduler = trainer.optimizers[0], trainer.lr_scheduler_configs[0].scheduler
        self._validate_optimizer_state(
            optimizer.param_groups, optimizer.state, scheduler.state_dict(), step, saved=False
        )
        if self.restored_step is None and (
            torch.count_nonzero(self.collaborative_residual.weight) or torch.count_nonzero(self.dynamic_gate.bias)
        ):
            raise ValueError("Unified scratch fresh fit requires the prescribed zero residual and gate initialization.")
        self._training_state_checked = True

    def on_train_batch_start(self, batch, batch_idx):
        if not self._training_state_checked or not 0 <= self.global_step < self.max_updates:
            raise ValueError("Unified scratch training requires an active fit within the 50000-update budget.")

    def on_save_checkpoint(self, checkpoint):
        step = checkpoint.get("global_step")
        if self.pretraining_metadata is not None or type(step) is not int or not 0 <= step <= self.max_updates:
            raise ValueError("Unified scratch cannot save external pretraining or an invalid update budget.")
        super().on_save_checkpoint(checkpoint)
        checkpoint["copmrec_unified_scratch"] = self.unified_scratch_contract

    def on_load_checkpoint(self, checkpoint):
        step = checkpoint.get("global_step")
        if (
            checkpoint.get("copmrec_unified_scratch") != self.unified_scratch_contract
            or type(step) is not int
            or not 0 <= step <= self.max_updates
            or checkpoint.get("copmrec_pretraining") is not None
        ):
            raise ValueError("Unified scratch checkpoint origin/seed/continuous-budget contract mismatch.")
        optimizers, schedulers = checkpoint.get("optimizer_states"), checkpoint.get("lr_schedulers")
        if (
            not isinstance(optimizers, list)
            or len(optimizers) != 1
            or not isinstance(optimizers[0], dict)
            or not isinstance(schedulers, list)
            or len(schedulers) != 1
        ):
            raise ValueError("Unified scratch normal restore requires complete optimizer and scheduler states.")
        self._validate_optimizer_state(
            optimizers[0].get("param_groups"), optimizers[0].get("state"), schedulers[0], step, saved=True
        )
        state = checkpoint.get("state_dict")
        expected = self.state_dict()
        if (
            not isinstance(state, dict)
            or set(state) != set(expected)
            or any(
                not isinstance(value, torch.Tensor)
                or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype
                or not torch.isfinite(value).all()
                for name, value in state.items()
            )
            or torch.count_nonzero(
                state["collaborative_residual.weight"][
                    ~self.seen_mask.to(state["collaborative_residual.weight"].device)
                ]
            )
        ):
            raise ValueError("Unified scratch checkpoint model state/cold residual contract mismatch.")
        super().on_load_checkpoint(checkpoint)
        self.restored_step = step
