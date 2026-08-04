from typing import Any

import torch
import torch.nn.functional as F
import transformers
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from src.common.components.eval_metrics import Evaluator
from src.common.configs.model import TrainingModelConfig
from src.data.components.data_models import (
    TigerLabelData,
    TigerModelInput,
)
from src.inference.model_output import ModelOutput
from src.recommendation.decoder import TigerDecoder
from src.recommendation.encoder import TigerEncoder
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
        huggingface_model: transformers.PreTrainedModel,
        decoder: transformers.PreTrainedModel,
        semantic_ids: torch.Tensor,
        num_hierarchies: int,
        codebook_width: int,
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
                "huggingface_model",
                "decoder",
                "semantic_ids",
                "training_model_config",
                "evaluator",
            ],
        )

        self.model = huggingface_model
        self.optimizer = training_model_config.optimizer
        self.scheduler = training_model_config.scheduler
        self.loss_function = training_model_config.loss_function
        self.evaluator = evaluator

        self.num_embeddings_per_hierarchy = codebook_width
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

        self.encoder = TigerEncoder(
            encoder=huggingface_model,
        )

        # bos_token used to prompt the decoder to generate the first token
        bos_token = nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)

        self.decoder = TigerDecoder(
            decoder=decoder,
            bos_token=bos_token,
            decoder_mlp=nn.ModuleList(
                [
                    nn.Linear(
                        self.embedding_dim,
                        self.num_embeddings_per_hierarchy,
                        bias=False,
                    )
                    for _ in range(self.num_hierarchies)
                ]
            ),
        )

        # generate embedding tables for each hierarchy
        # here we assume each hierarchy has the same amount of embeddings
        self.item_sid_embedding_table_encoder = nn.Embedding(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies,
            embedding_dim=self.embedding_dim,
        )

        # separation token for the encoder to differentiate between items
        self.sep_token = None
        if should_add_sep_token:
            self.sep_token = nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)

    def _inject_sep_token_between_sids(
        self,
        id_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sep_token: torch.Tensor,
        num_hierarchies: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject a separator token into the ID embeddings and attention mask."""
        batch_size, seq_len, emb_dim = id_embeddings.size()
        item_count_per_sequence = seq_len // num_hierarchies

        reshaped_id_embeddings = id_embeddings.view(batch_size, item_count_per_sequence, num_hierarchies, -1)
        reshaped_attention_mask = attention_mask.view(batch_size, item_count_per_sequence, num_hierarchies)
        reshaped_sep_token_for_concat = (
            sep_token.unsqueeze(0).expand(batch_size, item_count_per_sequence, -1).unsqueeze(-2)
        )
        id_embeddings = torch.cat([reshaped_id_embeddings, reshaped_sep_token_for_concat], dim=-2)
        attention_mask = torch.cat([reshaped_attention_mask, reshaped_attention_mask[:, :, [-1]]], dim=-1)
        id_embeddings = id_embeddings.reshape(batch_size, -1, emb_dim)
        attention_mask = attention_mask.reshape(batch_size, -1)
        return id_embeddings, attention_mask


    def _is_kv_cache_valid(self, kv_cache: tuple | DynamicCache | EncoderDecoderCache) -> bool:
        if isinstance(kv_cache, (EncoderDecoderCache, DynamicCache)):
            return len(kv_cache) > 0
        if isinstance(kv_cache, tuple):
            return True
        return False

    def _add_repeating_offset_to_rows(
        self,
        input_sids: torch.Tensor,
        codebook_size: int,
        num_hierarchies: int,
        attention_mask: torch.Tensor | None = None,
    ):
        """Add repeating hierarchy offsets to semantic IDs for a shared embedding table."""
        if input_sids.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional.")

        _, num_cols = input_sids.shape
        offsets = torch.arange(num_hierarchies, device=input_sids.device) * codebook_size
        num_repeats = (num_cols + num_hierarchies - 1) // num_hierarchies
        repeated_offsets = offsets.repeat(num_repeats)[:num_cols]

        input_sids_with_offsets = input_sids + repeated_offsets
        if attention_mask is not None:
            input_sids_with_offsets = input_sids_with_offsets * attention_mask
        return input_sids_with_offsets

    def _check_valid_prefix(self, prefix: torch.Tensor, batch_size: int = 100000) -> torch.Tensor:
        """Check if prefixes exist in the model-side semantic ID tensor."""
        if self.semantic_ids is None:
            raise ValueError("semantic_ids is required when should_check_prefix=True.")

        self.semantic_ids = self.semantic_ids.to(prefix.device)
        current_hierarchy = prefix.shape[1]
        num_prefixes = prefix.shape[0]
        results = []

        trimmed_semantic_ids = self.semantic_ids[:, :current_hierarchy]

        for i in range(0, num_prefixes, batch_size):
            batch_prefix = prefix[i: i + batch_size]
            comparison = trimmed_semantic_ids.unsqueeze(1) == batch_prefix.unsqueeze(0)
            all_match = comparison.all(dim=2)
            any_match = all_match.any(dim=0)
            results.append(any_match)

        return torch.cat(results)

    def _beam_search_one_step(
        self,
        candidate_logits: torch.Tensor,
        generated_ids: torch.Tensor | None,
        marginal_log_prob: torch.Tensor | None,
        past_key_values: EncoderDecoderCache | None,
        hierarchy: int,
        batch_size: int,
    ):
        """Perform one step of constrained beam search."""
        if self.should_check_prefix:
            if generated_ids is None:
                valid_prefix_mask = self._check_valid_prefix(
                    torch.arange(
                        self.num_embeddings_per_hierarchy,
                        device=candidate_logits.device,
                    ).unsqueeze(1)
                )
                candidate_logits[:, ~valid_prefix_mask] = float("-inf")
            else:
                valid_prefix_mask = self._check_valid_prefix(
                    torch.cat(
                        [
                            generated_ids.reshape(-1, hierarchy).repeat_interleave(
                                self.num_embeddings_per_hierarchy, dim=0
                            ),
                            torch.arange(
                                self.num_embeddings_per_hierarchy,
                                device=candidate_logits.device,
                            )
                            .repeat(self.top_k_for_generation * batch_size)
                            .unsqueeze(1),
                        ],
                        dim=1,
                    )
                ).reshape(-1, self.num_embeddings_per_hierarchy)
                candidate_logits[~valid_prefix_mask] = float("-inf")

        candidate_logits = F.softmax(candidate_logits, dim=-1)
        proba, indices = torch.sort(candidate_logits, descending=True)

        if generated_ids is None:
            proba_topk, indices_topk = (
                proba[:, : self.top_k_for_generation],
                indices[:, : self.top_k_for_generation],
            )
            generated_ids = indices_topk.unsqueeze(-1)
            self_attention_cache = DynamicCache()
            cross_attention_cache = DynamicCache()
            past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
            replace_indices = None
        else:
            proba, indices = (
                proba[:, : self.num_embeddings_per_hierarchy],
                indices[:, : self.num_embeddings_per_hierarchy],
            )
            proba, indices = (
                proba.reshape(-1, self.top_k_for_generation * self.num_embeddings_per_hierarchy),
                indices.reshape(-1, self.top_k_for_generation * self.num_embeddings_per_hierarchy),
            )
            proba = torch.mul(
                marginal_log_prob.repeat_interleave(self.num_embeddings_per_hierarchy, dim=-1),
                proba,
            )
            topk_results = torch.topk(torch.nan_to_num(proba, nan=-1), k=self.top_k_for_generation, dim=-1)
            proba_topk, indices_topk = topk_results.values, topk_results.indices
            replace_indices = (
                (indices_topk // self.num_embeddings_per_hierarchy)
                + torch.arange(indices_topk.size(0), device=proba.device).unsqueeze(1) * self.top_k_for_generation
            ).flatten()
            if past_key_values is not None:
                past_key_values.reorder_cache(replace_indices)

            indices_topk = torch.gather(indices, 1, indices_topk)

        if replace_indices is not None:
            generated_ids = torch.cat(
                [
                    generated_ids.reshape(-1, hierarchy)[replace_indices].reshape(
                        -1, self.top_k_for_generation, hierarchy
                    ),
                    indices_topk.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            generated_ids = indices_topk.unsqueeze(-1)

        return generated_ids, proba_topk, past_key_values

    def encoder_forward_pass(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder module.

        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
        """

        # we shift the IDs here to match the hierarchy structure
        # so that we can use a single embedding table to store the embeddings for all hierarchies
        shifted_sids = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds_for_encoder = self.item_sid_embedding_table_encoder(shifted_sids)

        if self.sep_token is not None:
            (
                inputs_embeds_for_encoder,
                attention_mask,
            ) = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds_for_encoder,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        attention_mask_for_encoder = attention_mask

        encoder_output = self.encoder(
            sequence_embedding=inputs_embeds_for_encoder,
            attention_mask=attention_mask_for_encoder,
        )
        return encoder_output, attention_mask_for_encoder

    def decoder_forward_pass(
        self,
        attention_mask: torch.Tensor | None = None,
        future_ids: torch.Tensor | None = None,
        encoder_output: torch.Tensor | None = None,
        attention_mask_for_encoder: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: DynamicCache | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            future_ids (torch.Tensor | None): The future IDs for the decoder.
            encoder_output (torch.Tensor | None): The output from the encoder.
            attention_mask_for_encoder (torch.Tensor | None): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (DynamicCache | None): The cache for past key values.
        """

        # we generated something before and we need to shift the future_ids
        if future_ids is not None:
            shifted_future_sids = self._add_repeating_offset_to_rows(
                input_sids=future_ids,
                codebook_size=self.num_embeddings_per_hierarchy,
                num_hierarchies=self.num_hierarchies,
                attention_mask=torch.ones_like(future_ids, device=future_ids.device)
                if attention_mask is None
                else attention_mask,
            )
            inputs_embeds_for_decoder = self.item_sid_embedding_table_encoder(shifted_future_sids)

            # we do not have valid kv cache
            # we need to prepend bos token to the decoder input
            if not self._is_kv_cache_valid(kv_cache=past_key_values):
                inputs_embeds_for_decoder = torch.cat(
                    [
                        self.decoder.bos_token.unsqueeze(0).expand(future_ids.size(0), 1, -1),
                        inputs_embeds_for_decoder,
                    ],
                    dim=1,
                )
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            torch.ones(future_ids.size(0), 1, device=future_ids.device),
                            attention_mask,
                        ],
                        dim=1,
                    )
            else:
                # we have valid kv cache
                # we only need the last token in the decoder input
                inputs_embeds_for_decoder = inputs_embeds_for_decoder[:, -1:, :]
        # this is the beginning of generation, we start from bos token
        else:
            inputs_embeds_for_decoder = self.decoder.bos_token.unsqueeze(0).expand(encoder_output.size(0), 1, -1)

        decoder_output = self.decoder(
            sequence_embedding=inputs_embeds_for_decoder,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask_for_encoder,
            encoder_output=encoder_output,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        return decoder_output

    def generate(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate the semantic id given the current model in the sequence using beam search.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
        """

        # getting encoder output
        # we only need to do this once because we have decoder
        # to do auto-regressive generation
        encoder_output, encoder_attention_mask = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
        )

        # initilize cached generated ids to None
        generated_ids = None
        marginal_log_prob = None

        # initialize kv cache
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()
        past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

        for hierarchy in range(self.num_hierarchies):
            if generated_ids is not None:
                # we generated something before
                # we need to reshape the generated ids so that
                # the number of beams equals to batch size * top_k
                squeezed_generated_ids = generated_ids.reshape(-1, hierarchy).to(
                    encoder_output.device
                )  # shape: (batch_size * top_k, hierarchy)

                repeated_encoder_output = encoder_output.repeat_interleave(self.top_k_for_generation, dim=0)
                # shape: (batch_size * top_k, seq_len, hidden_dim)

                repeated_encoder_attention_mask = encoder_attention_mask.repeat_interleave(
                    self.top_k_for_generation, dim=0
                )  # shape: (batch_size * top_k, seq_len+1)
            else:
                # we haven't generated anything yet!
                # the number of beams currently equals to batch size
                squeezed_generated_ids = None
                repeated_encoder_output = encoder_output
                repeated_encoder_attention_mask = encoder_attention_mask

            # feeding the decoder with the generated ids
            decoder_output, past_key_values = self.decoder_forward_pass(
                future_ids=squeezed_generated_ids,
                encoder_output=repeated_encoder_output,
                attention_mask_for_encoder=repeated_encoder_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

            # decoder_output[:, -1, :] is the embedding for the next token
            latest_output_representation = decoder_output[:, -1, :]

            # # calculating the logits for the next token
            candidate_logits = self.decoder.decoder_mlp[hierarchy](
                latest_output_representation
            )  # shape: (batch_size * top_k, num_embeddings in the hierarchy)

            (
                generated_ids,
                marginal_log_prob,
                past_key_values,
            ) = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                marginal_log_prob=marginal_log_prob,
                past_key_values=past_key_values,
                hierarchy=hierarchy,
                batch_size=input_ids.size(0),
            )

        return generated_ids, marginal_log_prob

    def forward(
        self,
        attention_mask_encoder: torch.Tensor,
        input_ids: torch.Tensor,
        future_ids: torch.Tensor | None = None,
        attention_mask_decoder: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder-decoder model.
        Parameters:
            attention_mask_encoder (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            future_ids (torch.Tensor | None): The future IDs for the decoder.
            attention_mask_decoder (torch.Tensor | None): The attention mask for the decoder.
        """

        encoder_output, attention_mask_for_encoder = self.encoder_forward_pass(
            attention_mask=attention_mask_encoder,
            input_ids=input_ids,
        )

        decoder_output = self.decoder_forward_pass(
            future_ids=future_ids,
            attention_mask=attention_mask_decoder,
            encoder_output=encoder_output,
            attention_mask_for_encoder=attention_mask_for_encoder,
            use_cache=False,  # we are not using cache for training
        )
        return decoder_output


    def predict_step(self, batch: TigerModelInput):
        generated_sids, _ = self.model_step(batch)
        if batch.output_keys is None:
            raise ValueError("TigerModelInput.output_keys is required for prediction output.")
        return ModelOutput(keys=batch.output_keys, predictions=generated_sids)

    def model_step(
        self,
        model_input: TigerModelInput,
        label_data: TigerLabelData | None = None,
    ):
        """
        Perform a forward pass of the model and calculate the loss if label_data is provided.

        Args:
            model_input: The input data to the model.
            label_data: The label data to the model. Its optional as it is not required for inference.
        """

        # if label_data is None, we are in inference mode and doing free-form generation
        if label_data is None:
            # this is inference stage
            generated_ids, marginal_probs = self.generate(
                attention_mask=model_input.attention_mask,
                input_ids=model_input.input_ids,
            )
            return generated_ids, 0  # returning 0 here because we don't have a loss

        fut_ids = label_data.target_ids
        # here we pass labels in to the forward function
        # because the decoder is causal and we are doing shifted prediction
        model_output = self.forward(
            attention_mask_encoder=model_input.attention_mask,
            input_ids=model_input.input_ids,
            future_ids=fut_ids,
        )

        # we prepended a bos token to the decoder input
        # so we need to remove the last token in the output
        model_output = model_output[:, :-1]

        loss = 0
        for hierarchy in range(self.num_hierarchies):
            input_ = self.decoder.decoder_mlp[hierarchy](model_output[:, hierarchy])
            loss += self.loss_function(
                input=input_,
                target=fut_ids[:, hierarchy].long(),
            )
        loss = loss / self.num_hierarchies
        return model_output, loss

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
        _, loss = self.model_step(model_input=model_input, label_data=label_data)

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
        _, loss = self.model_step(model_input=model_input, label_data=label_data)

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
        self._make_deterministic(is_training=True)

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

    def _make_deterministic(self, is_training: bool):
        """Set encoder and decoder training flags explicitly for generation stages."""
        if is_training:
            if self.decoder is not None:
                self.decoder.decoder.is_training = True
                self.decoder.decoder.train()
            if self.encoder is not None:
                self.encoder.encoder.is_training = True
                self.encoder.encoder.train()
        else:
            if self.decoder is not None:
                self.decoder.decoder.is_training = False
                self.decoder.decoder.eval()
            if self.encoder is not None:
                self.encoder.encoder.is_training = False
                self.encoder.encoder.eval()

    def on_predict_start(self):
        super().on_predict_start()
        self._make_deterministic(is_training=False)

    def on_predict_end(self):
        super().on_predict_end()
        self._make_deterministic(is_training=True)

    def on_validation_start(self):
        super().on_validation_start()
        self._make_deterministic(is_training=False)

    def on_validation_end(self):
        super().on_validation_end()
        self._make_deterministic(is_training=True)

    def on_test_start(self):
        super().on_test_start()
        self._make_deterministic(is_training=False)

    def on_test_end(self):
        super().on_test_end()
        self._make_deterministic(is_training=True)
