from typing import Optional

import torch
import transformers
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import Seq2SeqModelOutput

from src.utils.utils import delete_module, reset_parameters


class SemanticIDDecoderModule(torch.nn.Module):
    """
    This is an in-house replication of the decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        decoder: transformers.PreTrainedModel,
        decoder_mlp: Optional[torch.nn.Module] = None,
        bos_token: Optional[torch.nn.Parameter] = None,
    ):
        """
        Initialize the SemanticIDDecoderModule.

        Parameters:
        decoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        decoder_mlp (torch.nn.Module): the mlp layers used to project the decoder output to the embedding table.
        bos_token (Optional[torch.nn.Parameter]):
            the bos token used to prompt the decoder.
            if None, then this means the decoder is used standalone without an encoder.
        """

        super().__init__()
        # some sanity checks
        if bos_token is not None:
            assert decoder.config.is_decoder is True, "Decoder must be a decoder model"
            assert decoder.config.is_encoder_decoder is False, "Decoder must be a standalone decoder model"

        self.decoder = decoder
        # this bos token is prompt for the decoder
        self.bos_token = bos_token
        self.decoder_mlp = decoder_mlp
        # deleting embedding table in the decoder to save space
        delete_module(self.decoder, "embed_tokens")
        delete_module(self.decoder, "shared")
        reset_parameters(self.decoder)

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        use_cache: bool = False,
        past_key_values: DynamicCache = DynamicCache(),
    ):
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            sequence_embedding (torch.Tensor): The input sequence embedding for the decoder.
            encoder_output (torch.Tensor): The output from the encoder.
            encoder_attention_mask (torch.Tensor): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (DynamicCache): The cache for past key values.
        """

        decoder_outputs: Seq2SeqModelOutput = self.decoder(
            attention_mask=attention_mask,
            inputs_embeds=sequence_embedding,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        embeddings = decoder_outputs.last_hidden_state

        if use_cache:
            return embeddings, decoder_outputs.past_key_values
        return embeddings
