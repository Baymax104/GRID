"""只改变训练目标的独立探针；保留 Tiger 的评估和生成。"""

import torch

from src.data.components.artifacts import resolve_checkpoint_path
from src.recommendation.tiger.tiger import Tiger
from src.recommendation.tiger_training_probe.objective import ConditionalBranchObjective


class TigerTrainingProbe(Tiger):
    def __init__(
        self,
        *,
        statistics,
        initialization_checkpoint_path,
        arm,
        resume_checkpoint_path=None,
        layers=(2, 3),
        alpha=0.25,
        cap=2.0,
        wandb_entity=None,
        wandb_project=None,
        **kwargs,
    ):
        if resume_checkpoint_path is not None:
            raise ValueError("Training probe uses weight initialization; ckpt_path resume must be null.")
        if arm not in ("ce", "reweighted"):
            raise ValueError("arm must be ce or reweighted.")
        if not initialization_checkpoint_path:
            raise ValueError("initialization_checkpoint_path is required.")
        super().__init__(**kwargs)
        if self.decoder.prefix_allocation.enabled:
            raise ValueError("Training probe requires unchanged baseline decoding.")
        if not torch.equal(statistics["semantic_ids"].cpu(), self.semantic_ids.cpu()):
            raise ValueError("Statistics and model semantic IDs differ.")
        self.arm = arm
        self.statistics_metadata = statistics["metadata"]
        self.initialization_reference = initialization_checkpoint_path
        self.branch_objective = ConditionalBranchObjective(
            self.semantic_ids, statistics["expected_counts"], self.num_embeddings_per_hierarchy, layers, alpha, cap
        )
        resolved = resolve_checkpoint_path(
            initialization_checkpoint_path, default_entity=wandb_entity, default_project=wandb_project
        )
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=True)

    def training_step(self, batch, batch_idx):
        if self.arm == "ce":
            return super().training_step(batch, batch_idx)
        model_input, label_data = batch
        if label_data is None:
            raise ValueError("Training probe requires target labels.")
        logits = self.forward(
            attention_mask_encoder=model_input.attention_mask,
            input_ids=model_input.input_ids,
            future_ids=label_data.target_ids,
        )
        return {"loss": self.branch_objective(logits, label_data.target_ids)}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["training_frequency_probe"] = {
            "arm": self.arm,
            "initialization_reference": self.initialization_reference,
            "optimizer_initialization": "fresh",
            "statistics": self.statistics_metadata,
            **self.branch_objective.audit(),
        }
