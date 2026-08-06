import torch
import torch.nn.functional as F
import transformers
from torch import nn

from src.recommendation.tiger.utils import add_hierarchy_offsets
from src.utils.model import delete_module, reset_parameters


class TigerDecoder(torch.nn.Module):
    """
    This is an in-house replication of the decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        decoder: transformers.PreTrainedModel,
        embedding_dim: int,
        sid_embedding_table: nn.Embedding,
        codebook_size: int,
        num_hierarchies: int,
        top_k_for_generation: int,
        semantic_ids: torch.Tensor,
        should_check_prefix: bool,
    ):
        """
        Initialize the TigerDecoder module.

        Parameters:
        decoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        embedding_dim (int): decoder input embedding dimension.
        """

        super().__init__()
        assert decoder.config.is_decoder is True, "Decoder must be a decoder model"
        assert decoder.config.is_encoder_decoder is False, "Decoder must be a standalone decoder model"

        self.decoder = decoder
        # this bos token is prompt for the decoder
        self.bos_token = torch.nn.Parameter(torch.randn(1, embedding_dim), requires_grad=True)
        self.sid_embedding_table = sid_embedding_table
        self.codebook_size = codebook_size
        self.num_hierarchies = num_hierarchies
        self.top_k_for_generation = top_k_for_generation
        self.semantic_ids = semantic_ids[:, :num_hierarchies].long()
        self.should_check_prefix = should_check_prefix
        self.lm_head = nn.Linear(
            sid_embedding_table.embedding_dim,
            self.num_hierarchies * self.codebook_size,
            bias=False,
        )
        # deleting embedding table in the decoder to save space
        delete_module(self.decoder, "embed_tokens")
        delete_module(self.decoder, "shared")
        reset_parameters(self.decoder)


    def _check_valid_prefix(self, prefix: torch.Tensor, batch_size: int = 100000) -> torch.Tensor:
        """Check if prefixes exist in the model-side semantic ID tensor."""
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
        hierarchy: int,
        batch_size: int,
    ):
        """Perform one step of constrained beam search."""
        if self.should_check_prefix:
            if generated_ids is None:
                valid_prefix_mask = self._check_valid_prefix(
                    torch.arange(
                        self.codebook_size,
                        device=candidate_logits.device,
                    ).unsqueeze(1)
                )
                candidate_logits[:, ~valid_prefix_mask] = float("-inf")
            else:
                valid_prefix_mask = self._check_valid_prefix(
                    torch.cat(
                        [
                            generated_ids.reshape(-1, hierarchy).repeat_interleave(
                                self.codebook_size, dim=0
                            ),
                            torch.arange(
                                self.codebook_size,
                                device=candidate_logits.device,
                            )
                            .repeat(self.top_k_for_generation * batch_size)
                            .unsqueeze(1),
                        ],
                        dim=1,
                    )
                ).reshape(-1, self.codebook_size)
                candidate_logits[~valid_prefix_mask] = float("-inf")

        candidate_logits = F.softmax(candidate_logits, dim=-1)
        proba, indices = torch.sort(candidate_logits, descending=True)

        if generated_ids is None:
            proba_topk, indices_topk = (
                proba[:, : self.top_k_for_generation],
                indices[:, : self.top_k_for_generation],
            )
            generated_ids = indices_topk.unsqueeze(-1)
            replace_indices = None
        else:
            proba, indices = (
                proba[:, : self.codebook_size],
                indices[:, : self.codebook_size],
            )
            proba, indices = (
                proba.reshape(-1, self.top_k_for_generation * self.codebook_size),
                indices.reshape(-1, self.top_k_for_generation * self.codebook_size),
            )
            proba = torch.mul(
                marginal_log_prob.repeat_interleave(self.codebook_size, dim=-1),
                proba,
            )
            topk_results = torch.topk(torch.nan_to_num(proba, nan=-1), k=self.top_k_for_generation, dim=-1)
            proba_topk, indices_topk = topk_results.values, topk_results.indices
            replace_indices = (
                (indices_topk // self.codebook_size)
                + torch.arange(indices_topk.size(0), device=proba.device).unsqueeze(1) * self.top_k_for_generation
            ).flatten()

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

        return generated_ids, proba_topk

    def forward(
        self,
        future_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run teacher-forcing decoder forward with BOS plus future SID inputs.

        Args:
            future_ids: Future semantic IDs with shape
                ``(batch_size, sequence_length)``.
            encoder_output: Encoder hidden states.
            encoder_attention_mask: Encoder attention mask.

        Returns:
            Raw semantic-ID logits with shape
            ``(batch_size, sequence_length, num_hierarchies * codebook_size)``.
        """

        shifted_sids = add_hierarchy_offsets(
            semantic_ids=future_ids,
            codebook_size=self.codebook_size,
            num_hierarchies=self.num_hierarchies,
        )
        sequence_embedding = self.sid_embedding_table(shifted_sids)

        sequence_embedding = torch.cat(
            [
                self.bos_token.unsqueeze(0).expand(future_ids.size(0), 1, -1),
                sequence_embedding,
            ],
            dim=1,
        )

        decoder_outputs = self.decoder(
            inputs_embeds=sequence_embedding,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
        )

        # Shape (batch_size, num_hierarchies, embedding_dim)
        decoder_hidden_states = decoder_outputs.last_hidden_state[:, :-1, :]
        return self.lm_head(decoder_hidden_states)

    def generate(
        self,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        batch_size: int,
    ):
        """Generate semantic IDs using autoregressive beam search."""
        generated_ids = None
        marginal_log_prob = None

        for hierarchy in range(self.num_hierarchies):
            if generated_ids is not None:
                squeezed_generated_ids = generated_ids.reshape(-1, hierarchy).to(encoder_output.device)
                repeated_encoder_output = encoder_output.repeat_interleave(self.top_k_for_generation, dim=0)
                repeated_encoder_attention_mask = encoder_attention_mask.repeat_interleave(
                    self.top_k_for_generation, dim=0
                )
                shifted_sids = add_hierarchy_offsets(
                    semantic_ids=squeezed_generated_ids,
                    codebook_size=self.codebook_size,
                    num_hierarchies=self.num_hierarchies,
                    attention_mask=torch.ones_like(squeezed_generated_ids, device=squeezed_generated_ids.device),
                )
                sequence_embedding = self.sid_embedding_table(shifted_sids)
                bos = self.bos_token.unsqueeze(0).expand(squeezed_generated_ids.size(0), 1, -1)
                sequence_embedding = torch.cat([bos, sequence_embedding], dim=1)
            else:
                repeated_encoder_output = encoder_output
                repeated_encoder_attention_mask = encoder_attention_mask
                sequence_embedding = self.bos_token.unsqueeze(0).expand(encoder_output.size(0), 1, -1)

            decoder_outputs = self.decoder(
                inputs_embeds=sequence_embedding,
                encoder_hidden_states=repeated_encoder_output,
                encoder_attention_mask=repeated_encoder_attention_mask,
                use_cache=False,
            )

            # Shape (batch_size, embedding_dim)
            latest_output_representation = decoder_outputs.last_hidden_state[:, -1, :]
            # Shape (batch_size, num_hierarchies * codebook_size)
            global_logits = self.lm_head(latest_output_representation)
            start_index = hierarchy * self.codebook_size
            end_index = start_index + self.codebook_size
            # Shape (batch_size, codebook_size)
            candidate_logits = global_logits[:, start_index:end_index]

            generated_ids, marginal_log_prob = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                marginal_log_prob=marginal_log_prob,
                hierarchy=hierarchy,
                batch_size=batch_size,
            )

        return generated_ids, marginal_log_prob
