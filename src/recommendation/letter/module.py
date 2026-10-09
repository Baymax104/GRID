"""独立LETTER的Lightning接口与严格checkpoint身份。"""

from lightning import LightningModule
from omegaconf import OmegaConf

from src.common.configs.model import TrainingModelConfig
from src.data.components.data_models import ModelOutput
from src.recommendation.letter.backbone import LetterBackbone


class LetterRecommender(LightningModule):
    def __init__(self, catalog, training_model_config=None, max_history_items=20, **backbone_kwargs):
        super().__init__()
        if max_history_items < 1:
            raise ValueError("LETTER history length must be positive.")
        self.training_model_config = training_model_config or TrainingModelConfig()
        backbone_kwargs = OmegaConf.to_container(OmegaConf.create(backbone_kwargs), resolve=True)
        self.identity = {
            "catalog_sha256": catalog.sha256,
            "codebook_size": catalog.codebook_size,
            "base_vocab_size": catalog.base_vocab_size,
            "max_history_items": max_history_items,
            "backbone": backbone_kwargs,
        }
        self.backbone = LetterBackbone(
            catalog.keys,
            catalog.semantic_ids,
            codebook_size=catalog.codebook_size,
            base_vocab_size=catalog.base_vocab_size,
            **backbone_kwargs,
        )

    def training_step(self, batch, batch_idx):
        loss, _ = self.backbone(batch["input_ids"], batch["attention_mask"], batch["labels"])
        return {"loss": loss}

    def evaluation_payload(self, batch):
        ids, scores = self.backbone.generate(batch["input_ids"], batch["attention_mask"])
        return {"generated_ids": ids, "scores": scores, "labels": batch["target"], "user_count": len(ids)}

    def validation_step(self, batch, batch_idx):
        return self.evaluation_payload(batch)

    def test_step(self, batch, batch_idx):
        return self.evaluation_payload(batch)

    def predict_step(self, batch, batch_idx=0):
        ids, _ = self.backbone.generate(batch["input_ids"], batch["attention_mask"])
        # 公共本地writer按rank合并pickle，输出不能保留各rank的CUDA device。
        return ModelOutput(keys=batch["user_id"].detach().cpu(), predictions=ids.detach().cpu())

    def configure_optimizers(self):
        if self.training_model_config.optimizer is None:
            raise ValueError("LETTER optimizer must be configured.")
        # 对齐作者HF Trainer：bias与LayerNorm不使用weight decay。
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            (no_decay if parameter.ndim == 1 or name.endswith("bias") else decay).append(parameter)
        optimizer = self.training_model_config.optimizer(
            params=[{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
        )
        if self.training_model_config.scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.training_model_config.scheduler(optimizer=optimizer),
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["letter_identity"] = self.identity

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("letter_identity") != self.identity:
            raise ValueError("LETTER checkpoint catalog or architecture identity mismatch.")
