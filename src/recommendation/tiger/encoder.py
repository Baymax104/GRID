import torch
import transformers
from torch import nn

from src.recommendation.tiger.utils import (
    add_hierarchy_offsets,
    insert_separator_token_between_items,
)
from src.utils.model import delete_module, reset_parameters


class TigerEncoder(torch.nn.Module):
    """
    This is an in-house replication of the encoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        encoder: transformers.PreTrainedModel,
        sid_embedding_table: nn.Embedding,
        codebook_size: int,
        num_hierarchies: int,
        should_add_sep_token: bool = True,
    ):
        """
        Initialize the TigerEncoder module.

        Parameters:
            encoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        """
        super().__init__()
        self.encoder = encoder
        self.sid_embedding_table = sid_embedding_table
        self.codebook_size = codebook_size
        self.num_hierarchies = num_hierarchies
        self.sep_token = None
        if should_add_sep_token:
            self.sep_token = nn.Parameter(torch.randn(1, sid_embedding_table.embedding_dim), requires_grad=True)

        # deleting embedding table in the encoder to save space
        delete_module(self.encoder, "embed_tokens")
        delete_module(self.encoder, "shared")
        reset_parameters(self.encoder)

    def embed_semantic_ids(
        self,
        semantic_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shifted_sids = add_hierarchy_offsets(
            semantic_ids=semantic_ids,
            codebook_size=self.codebook_size,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        return self.sid_embedding_table(shifted_sids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_embedding = self.embed_semantic_ids(
            semantic_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.sep_token is not None:
            sequence_embedding, attention_mask = insert_separator_token_between_items(
                id_embeddings=sequence_embedding,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        encoder_output = self.encoder(
            inputs_embeds=sequence_embedding,
            attention_mask=attention_mask,
        )
        embeddings = encoder_output.last_hidden_state
        return embeddings, attention_mask
