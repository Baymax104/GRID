from typing import Any

import torch
import transformers
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric

from src.common.components.eval_metrics import Evaluator
from src.common.configs.model import TrainingModelConfig
from src.data.components.data_models import (
    TigerLabelData,
    TigerModelInput,
)
from src.inference.model_output import ModelOutput
from src.recommendation.tiger.decoder import TigerDecoder
from src.recommendation.tiger.encoder import TigerEncoder
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class Tiger(LightningModule):
    """
    This is an in-house implementation of the encoder-decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    We added some additional features and modifications to the original architecture.
    (e.g., constrained beam search, separation tokens, etc)
    """

    def __init__(
        self,
        encoder: transformers.PreTrainedModel,
        decoder: transformers.PreTrainedModel,
        semantic_ids: torch.Tensor,
        num_hierarchies: int,
        codebook_size: int,
        embedding_dim: int,
        top_k_for_generation: int = 10,
        should_check_prefix: bool = False,
        should_add_sep_token: bool = True,
        training_model_config: TrainingModelConfig | None = None,
        evaluator: Evaluator | None = None,
    ):
        super().__init__()

        if training_model_config is None:
            training_model_config = TrainingModelConfig()

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "encoder",
                "decoder",
                "semantic_ids",
                "training_model_config",
                "evaluator",
            ],
        )

        self.optimizer = training_model_config.optimizer
        self.scheduler = training_model_config.scheduler
        self.loss_function = training_model_config.loss_function
        self.evaluator = evaluator

        self.num_embeddings_per_hierarchy = codebook_size
        self.embedding_dim = embedding_dim
        self.num_hierarchies = num_hierarchies
        self.should_check_prefix = should_check_prefix
        self.top_k_for_generation = top_k_for_generation
        if semantic_ids.ndim != 2:
            raise ValueError(
                f"semantic_ids must have shape (num_items, num_hierarchies), got {tuple(semantic_ids.shape)}."
            )
        if semantic_ids.size(1) < num_hierarchies:
            raise ValueError(
                f"semantic_ids second dimension ({semantic_ids.size(1)}) must be >= num_hierarchies ({num_hierarchies})."
            )
        self.semantic_ids = semantic_ids[:, :num_hierarchies].long()

        if self.evaluator:  # For inference, evaluator is not set.
            for metric_name, metric_object in self.evaluator.metrics.items():
                setattr(self, metric_name, metric_object)

            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()
            self.test_loss = MeanMetric()

        self.sid_embedding_table = nn.Embedding(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies,
            embedding_dim=self.embedding_dim,
        )

        self.encoder = TigerEncoder(
            encoder=encoder,
            sid_embedding_table=self.sid_embedding_table,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            should_add_sep_token=should_add_sep_token,
        )

        self.decoder = TigerDecoder(
            decoder=decoder,
            embedding_dim=self.embedding_dim,
            sid_embedding_table=self.sid_embedding_table,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            top_k_for_generation=self.top_k_for_generation,
            semantic_ids=self.semantic_ids,
            should_check_prefix=self.should_check_prefix,
        )

    def generate(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ):
        """
        Generate the semantic id given the current model in the sequence using beam search.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
        """

        # getting encoder output
        # we only need to do this once because we have decoder
        # to do auto-regressive generation
        encoder_output, encoder_attention_mask = self.encoder(
            attention_mask=attention_mask,
            input_ids=input_ids,
        )

        return self.decoder.generate(
            encoder_output=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            batch_size=input_ids.size(0),
        )

    def forward(
        self,
        attention_mask_encoder: torch.Tensor,
        input_ids: torch.Tensor,
        future_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run teacher-forcing encoder-decoder computation and return raw logits.

        Args:
            attention_mask_encoder: The attention mask for the encoder.
            input_ids: The input IDs for the encoder.
            future_ids: The future semantic IDs for the decoder.

        Returns:
            Raw semantic-ID logits with shape
            ``(batch_size, sequence_length, num_hierarchies * codebook_size)``.
        """

        encoder_output, attention_mask_for_encoder = self.encoder(
            attention_mask=attention_mask_encoder,
            input_ids=input_ids,
        )

        decoder_logits = self.decoder(
            future_ids=future_ids,
            encoder_output=encoder_output,
            encoder_attention_mask=attention_mask_for_encoder,
        )
        return decoder_logits

    def predict_step(self, batch: TigerModelInput):
        generated_sids, _ = self.generate(
            attention_mask=batch.attention_mask,
            input_ids=batch.input_ids,
        )
        if batch.output_keys is None:
            raise ValueError("TigerModelInput.output_keys is required for prediction output.")
        return ModelOutput(keys=batch.output_keys, predictions=generated_sids)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-hierarchy TIGER loss from teacher-forcing logits.

        Args:
            logits: Raw semantic-ID logits with shape
                ``(batch_size, num_hierarchies, num_hierarchies * codebook_size)``.
            target_ids: Target semantic IDs with shape
                ``(batch_size, num_hierarchies)``.
        """

        hierarchy_offsets = torch.arange(
            self.num_hierarchies,
            device=target_ids.device,
        ) * self.num_embeddings_per_hierarchy
        global_target_ids = target_ids + hierarchy_offsets

        loss = 0
        for hierarchy in range(self.num_hierarchies):
            loss += self.loss_function(
                input=logits[:, hierarchy],
                target=global_target_ids[:, hierarchy].long(),
            )
        loss = loss / self.num_hierarchies
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and optional step scheduler for Lightning."""
        if self.optimizer is None:
            raise ValueError("optimizer is required for training.")

        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def log_metrics(
        self,
        prefix: str,
        on_step=False,
        on_epoch=True,
        sync_dist=False,
        logger=True,
        prog_bar=False,
        call_compute=False,
    ):
        if self.evaluator is None:
            return

        metrics_dict = {
            f"{prefix}/{metric_name}": metric_object.compute() if call_compute else metric_object
            for metric_name, metric_object in self.evaluator.metrics.items()
        }

        self.log_dict(
            metrics_dict,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            logger=logger,
            prog_bar=prog_bar,
        )

    def training_step(
        self,
        batch: tuple[TigerModelInput, TigerLabelData | None],
        batch_idx: int,
    ) -> torch.Tensor:
        model_input, label_data = batch
        if label_data is None:
            raise ValueError("Tiger training_step requires label_data.")

        target_ids = label_data.target_ids
        logits = self.forward(
            attention_mask_encoder=model_input.attention_mask,
            input_ids=model_input.input_ids,
            future_ids=target_ids,
        )
        loss = self._compute_loss(logits=logits, target_ids=target_ids)

        if self.evaluator:
            self.train_loss(loss)
            self.log(
                "train/loss",
                self.train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        return loss

    def eval_step(
        self,
        batch: tuple[TigerModelInput, TigerLabelData],
        loss_to_aggregate: MeanMetric,
    ):
        """Perform a TIGER generation evaluation step."""
        if self.evaluator is None:
            return

        model_input: TigerModelInput = batch[0]
        label_data: TigerLabelData = batch[1]
        target_ids = label_data.target_ids
        logits = self.forward(
            attention_mask_encoder=model_input.attention_mask,
            input_ids=model_input.input_ids,
            future_ids=target_ids,
        )
        loss = self._compute_loss(logits=logits, target_ids=target_ids)

        # Evaluation reports both teacher-forcing loss and autoregressive ranking metrics.
        generated_ids, marginal_probs = self.generate(
            attention_mask=model_input.attention_mask,
            input_ids=model_input.input_ids,
        )

        self.evaluator(
            marginal_probs=marginal_probs,
            generated_ids=generated_ids,
            labels=label_data.target_ids.to(marginal_probs.device),
        )

        loss_to_aggregate(loss)

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ):
        if self.evaluator is None:
            return
        self.eval_step(batch, self.val_loss)

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
    ):
        if self.evaluator is None:
            return
        self.eval_step(batch, self.test_loss)

    def on_train_start(self):
        super().on_train_start()
        if self.evaluator:
            self.val_loss.reset()
            self.evaluator.reset()
            self.train_loss.reset()
            self.test_loss.reset()

    def on_validation_epoch_start(self):
        if self.evaluator:
            self.val_loss.reset()
            self.evaluator.reset()

    def on_test_epoch_start(self):
        if self.evaluator:
            self.test_loss.reset()
            self.evaluator.reset()

    def on_validation_epoch_end(self):
        if self.evaluator:
            self.log("val/loss", self.val_loss, sync_dist=False, prog_bar=False, logger=True)
            self.log_metrics("val")

    def on_test_epoch_end(self):
        if self.evaluator:
            self.log("test/loss", self.test_loss, sync_dist=False, prog_bar=False, logger=True)
            self.log_metrics("test")

    def on_exception(self, exception):
        self.trainer.should_stop = True
        if self.trainer.logger is not None:
            self.trainer.logger.finalize(status="failure")
