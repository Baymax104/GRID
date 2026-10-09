"""独立LETTER tokenizer的训练、全目录碰撞评价和SID输出。"""

import hashlib

from lightning import LightningModule
from omegaconf import OmegaConf

from src.common.configs.model import TrainingModelConfig
from src.data.components.data_models import ModelOutput
from src.quantization.letter.tokenizer import LetterTokenizer


class LetterTokenizerModule(LightningModule):
    def __init__(self, embeddings, training_model_config=None, seed=42, **tokenizer_kwargs):
        super().__init__()
        if tokenizer_kwargs.get("num_layers", 4) != 4 or tokenizer_kwargs.get("latent_dim", 32) != 32:
            raise ValueError("Formal LETTER tokenizer requires four levels and latent dimension 32.")
        self.training_model_config = training_model_config or TrainingModelConfig()
        tokenizer_kwargs = OmegaConf.to_container(OmegaConf.create(tokenizer_kwargs), resolve=True)
        self.embeddings, self.seed = embeddings, seed
        digest = hashlib.sha256()
        for name in ("keys", "content", "cf"):
            digest.update(embeddings[name].contiguous().numpy().tobytes())
        self.identity = {"input_sha256": digest.hexdigest(), "tokenizer": tokenizer_kwargs}
        self.backbone = LetterTokenizer(**tokenizer_kwargs)

    def on_fit_start(self):
        if self.trainer.world_size != 1:
            raise ValueError("LETTER tokenizer uses full-catalog initialization and requires a single GPU.")
        if not self.backbone.initialized:
            self.backbone.initialize(self.embeddings["content"].to(self.device), self.seed)

    def on_train_epoch_start(self):
        if self.current_epoch > 0:
            self.backbone.update_groups(self.seed + self.current_epoch)

    def training_step(self, batch, batch_idx):
        return self.backbone(batch[1], batch[2])

    def validation_step(self, batch, batch_idx):
        codes = self.backbone.encode(batch[1])
        return {"collision_rate": 1 - codes.unique(dim=0).shape[0] / len(codes)}

    def predict_step(self, batch, batch_idx=0):
        if len(batch[0]) != len(self.embeddings["keys"]):
            raise ValueError("LETTER SID export requires the entire catalog in one batch.")
        return ModelOutput(keys=batch[0].detach().cpu(), predictions=self.backbone.unique_codes(batch[1]).cpu())

    def configure_optimizers(self):
        if self.training_model_config.optimizer is None:
            raise ValueError("LETTER tokenizer optimizer must be configured.")
        return self.training_model_config.optimizer(params=self.parameters())

    def on_save_checkpoint(self, checkpoint):
        checkpoint["letter_tokenizer_identity"] = self.identity

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("letter_tokenizer_identity") != self.identity:
            raise ValueError("LETTER tokenizer checkpoint input or architecture mismatch.")
