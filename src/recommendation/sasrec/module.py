"""SASRec 的 Lightning 训练与共同全目录评价接口。"""

from typing import Any

import torch
from lightning import LightningModule

from src.common.configs.model import TrainingModelConfig
from src.data.components.data_models import ModelOutput, SASRecLabelData, SASRecModelInput
from src.data.components.sasrec import ItemCatalog
from src.recommendation.sasrec.backbone import OFFICIAL_COMMIT, SASRecBackbone


class SASRec(LightningModule):
    def __init__(
        self,
        catalog: ItemCatalog,
        max_history_items: int = 50,
        hidden_size: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.5,
        l2_emb: float = 0.0,
        top_k: int = 10,
        catalog_chunk_size: int = 4096,
        training_model_config: TrainingModelConfig | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= len(catalog) or catalog_chunk_size < 1:
            raise ValueError("top_k must fit the catalog and chunk_size must be positive.")
        self.register_buffer("item_keys", catalog.keys.clone())
        self.catalog_sha256 = catalog.sha256
        self.top_k = top_k
        self.catalog_chunk_size = catalog_chunk_size
        self.training_model_config = training_model_config or TrainingModelConfig()
        self.backbone = SASRecBackbone(
            len(catalog), max_history_items, hidden_size, num_blocks, num_heads, dropout, l2_emb
        )
        self.protocol = {
            "official_commit": OFFICIAL_COMMIT,
            "algorithm": "official-q-only-ln-pointwise-bce-v1",
            "max_history_items": max_history_items,
            "hidden_size": hidden_size,
            "num_blocks": num_blocks,
            "num_heads": num_heads,
            "dropout": dropout,
            "l2_emb": l2_emb,
        }

    def training_step(self, batch: tuple[SASRecModelInput, SASRecLabelData], batch_idx: int) -> dict:
        model_input, labels = batch
        if labels.negative_ids is None:
            raise ValueError("SASRec training requires per-position negative IDs.")
        positive, negative = self.backbone(model_input.input_ids, labels.target_ids, labels.negative_ids)
        return {"loss": self.backbone.objective(positive, negative, labels.target_ids)}

    def configure_optimizers(self) -> Any:
        if self.training_model_config.optimizer is None:
            raise ValueError("SASRec optimizer must be configured for training.")
        optimizer = self.training_model_config.optimizer(params=self.parameters())
        if self.training_model_config.scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.training_model_config.scheduler(optimizer=optimizer),
                "interval": "step",
            },
        }

    def retrieve(self, model_input: SASRecModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.backbone.encode(model_input.input_ids)[:, -1]
        batch_size = query.shape[0]
        best_scores = query.new_empty(batch_size, 0)
        best_ids = torch.empty(batch_size, 0, device=query.device, dtype=torch.long)
        for start in range(1, len(self.item_keys) + 1, self.catalog_chunk_size):
            ids = torch.arange(
                start, min(start + self.catalog_chunk_size, len(self.item_keys) + 1), device=query.device
            )
            scores = self.backbone.score_items(query, ids)
            scores = torch.cat((best_scores, scores), dim=1)
            ids = torch.cat((best_ids, ids.expand(batch_size, -1)), dim=1)
            # 已保留候选内部同分按 key 升序，新块 key 更大，stable 排序保持该顺序。
            order = scores.argsort(dim=1, descending=True, stable=True)[:, : self.top_k]
            best_scores = scores.gather(1, order)
            best_ids = ids.gather(1, order)
        return self.item_keys[best_ids - 1], best_scores

    def evaluation_payload(self, batch: tuple[SASRecModelInput, SASRecLabelData]) -> dict:
        model_input, labels = batch
        if labels.target_ids.ndim != 1 or labels.negative_ids is not None:
            raise ValueError("SASRec evaluation requires one target per user and no negatives.")
        self.backbone._validate_ids(labels.target_ids, "target_ids")
        if (labels.target_ids == 0).any():
            raise ValueError("Evaluation targets cannot be padding.")
        predictions, scores = self.retrieve(model_input)
        return {
            "generated_ids": predictions,
            "scores": scores,
            "labels": self.item_keys[labels.target_ids.long() - 1],
            "user_count": len(labels.target_ids),
        }

    def validation_step(self, batch, batch_idx):
        return self.evaluation_payload(batch)

    def test_step(self, batch, batch_idx):
        return self.evaluation_payload(batch)

    def predict_step(self, batch, batch_idx=0):
        model_input = batch if isinstance(batch, SASRecModelInput) else batch[0]
        if model_input.output_keys is None:
            raise ValueError("SASRec prediction requires output user keys.")
        predictions, _ = self.retrieve(model_input)
        return ModelOutput(keys=model_input.output_keys, predictions=predictions)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["sasrec_identity"] = {"catalog_sha256": self.catalog_sha256, **self.protocol}

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        expected = {"catalog_sha256": self.catalog_sha256, **self.protocol}
        if checkpoint.get("sasrec_identity") != expected:
            raise ValueError("SASRec checkpoint catalog or algorithm/training structure identity mismatch.")
        saved_keys = checkpoint.get("state_dict", {}).get("item_keys")
        if saved_keys is None or not torch.equal(saved_keys.detach().cpu(), self.item_keys.detach().cpu()):
            raise ValueError("SASRec checkpoint item-key mapping mismatch.")
