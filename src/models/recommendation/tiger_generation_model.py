from typing import Any, Optional

import torch
import transformers
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.common.components.model_output import OneKeyPerPredictionOutput
from src.models.recommendation.modules import (
    SemanticIDDecoderModule,
    SemanticIDEncoderModule,
    SemanticIDGenerativeRecommender,
    T5MultiLayerFF,
)
from src.utils.utils import get_parent_module_and_attr


class SemanticIDEncoderDecoder(SemanticIDGenerativeRecommender):
    """
    This is an in-house implementation of the encoder-decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    We added some additional features and modifications to the original architecture.
    (e.g., constrained beam search, separation tokens, etc)
    """

    def __init__(
        self,
        top_k_for_generation: int = 10,
        codebooks: torch.Tensor = None,
        embedding_dim: int = None,
        num_hierarchies: int = None,
        num_embeddings_per_hierarchy: int = None,
        num_user_bins: Optional[int] = None,
        mlp_layers: Optional[int] = None,
        should_check_prefix: bool = False,
        should_add_sep_token: bool = True,
        prediction_key_name: str = "user_id",
        prediction_value_name: str = "semantic_ids",
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDEncoderDecoder module.

        Paremeters:
        codebooks (torch.Tensor): the codebooks for the semantic ID.
            the shape of the codebooks should be (num_hierarchies, num_embeddings_per_hierarchy).
        num_hierarchies (int): the number of hierarchies in the codebooks.
        top_k_for_generation (int): the number of top-k candidates for generation.
        num_user_bins (Optional[int]): the number of bins for user in the dataset (this number equals to the number of rows in the embedding table ).
        mlp_layers (Optional[int]): the number of mlp layers in the encoder and decoder.
        embedding_dim (Optional[int]): the dimension of the embeddings.
        should_check_prefix (bool): whether to check if the prefix is valid.
        """

        if num_hierarchies is None or num_embeddings_per_hierarchy is None:
            num_hierarchies, num_embeddings_per_hierarchy = (
                codebooks.shape[0],
                codebooks.max().item() + 1,
            )
        if embedding_dim is None:
            embedding_dim = (
                kwargs["huggingface_model"]
                .encoder
                .block[0]
                .layer[0]
                .SelfAttention.q.in_features
            )

        super().__init__(
            codebooks=codebooks,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            embedding_dim=embedding_dim,
            top_k_for_generation=top_k_for_generation,
            should_check_prefix=should_check_prefix,
            **kwargs,
        )

        self.encoder = SemanticIDEncoderModule(
            encoder=self.encoder,
        )

        # bos_token used to prompt the decoder to generate the first token
        bos_token = torch.nn.Parameter(
            torch.randn(1, self.embedding_dim), requires_grad=True
        )

        self.decoder = SemanticIDDecoderModule(
            decoder=self.decoder,
            bos_token=bos_token,
            decoder_mlp=torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.embedding_dim,
                        self.num_embeddings_per_hierarchy,
                        bias=False,
                    )
                    for _ in range(self.num_hierarchies)
                ]
            ),
        )

        if mlp_layers is not None:
            # bloating the mlp layers in both encoder and decoder
            # TODO (clark): this currently only works for T5
            for name, module in self.named_modules():
                if isinstance(module, transformers.models.t5.modeling_t5.T5LayerFF):
                    parent_module, attr_name = get_parent_module_and_attr(self, name)
                    setattr(
                        parent_module,
                        attr_name,
                        T5MultiLayerFF(
                            config=self.encoder.encoder.config, num_layers=mlp_layers
                        ),
                    )

        # generate embedding tables for each hierarchy
        # here we assume each hierarchy has the same amount of embeddings
        self.item_sid_embedding_table_encoder = self._spawn_embedding_tables(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies,
            embedding_dim=self.embedding_dim,
        )

        # generating user embedding table
        self.user_embedding: torch.nn.Embedding = (
            self._spawn_embedding_tables(
                num_embeddings=num_user_bins,
                embedding_dim=self.embedding_dim,
            )
            if num_user_bins
            else None
        )

        # separation token for the encoder to differentiate between items
        self.sep_token = (
            torch.nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)
            if should_add_sep_token
            else None
        )
        # the key value names for the prediction output
        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name

    def encoder_forward_pass(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder module.

        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
        """

        # we shift the IDs here to match the hierarchy structure
        # so that we can use a single embedding table to store the embeddigns for all hierarchies
        shifted_sids = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds_for_encoder = self.get_embedding_table(table_name="encoder")(
            shifted_sids
        )

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

        # we enter this loop if we want to use user_id
        if user_id is not None and self.user_embedding is not None:
            # preprocessing function pad user_id with zeros
            # so we only need to take the first column
            user_id = user_id[:, 0]

            # TODO (clark): here we assume remainder hashing, which is different from LSH hashing used in TIGER.
            user_embeds = self.user_embedding(
                torch.remainder(user_id, self.user_embedding.num_embeddings)
            )

            # prepending the user_id embedding to the input senquence
            inputs_embeds_for_encoder = torch.cat(
                [
                    user_embeds.unsqueeze(1),
                    inputs_embeds_for_encoder,
                ],
                dim=1,
            )
            # prepending 1 to attention mask as we introduce user embedding in the first column
            user_attention_mask = torch.ones(
                attention_mask.size(0), 1, device=attention_mask.device
            )
            attention_mask_for_encoder = torch.cat(
                [
                    user_attention_mask,
                    attention_mask,
                ],
                dim=1,
            )
        else:
            attention_mask_for_encoder = attention_mask

        encoder_output = self.encoder(
            sequence_embedding=inputs_embeds_for_encoder,
            attention_mask=attention_mask_for_encoder,
        )
        return encoder_output, attention_mask_for_encoder

    def decoder_forward_pass(
        self,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # TODO (clark): in the future we should support variable length semantic id
        future_ids: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attention_mask_for_encoder: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[DynamicCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            future_ids (Optional[torch.Tensor]): The future IDs for the decoder.
            encoder_output (Optional[torch.Tensor]): The output from the encoder.
            attention_mask_for_encoder (Optional[torch.Tensor]): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (Optional[DynamicCache]): The cache for past key values.
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
            inputs_embeds_for_decoder = self.get_embedding_table(table_name="decoder")(
                shifted_future_sids
            )

            # we do not have valid kv cache
            # we need to prepend bos token to the decoder input
            if not self._is_kv_cache_valid(kv_cache=past_key_values):
                inputs_embeds_for_decoder = torch.cat(
                    [
                        self.decoder.bos_token.unsqueeze(0).expand(
                            future_ids.size(0), 1, -1
                        ),
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
            inputs_embeds_for_decoder = self.decoder.bos_token.unsqueeze(0).expand(
                encoder_output.size(0), 1, -1
            )

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
        user_id: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate the semantic id given the current model in the sequence using beam search.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
        """

        # getting encoder output
        # we only need to do this once because we have decoder
        # to do auto-regressive generation
        encoder_output, encoder_attention_mask = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
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

                repeated_encoder_output = encoder_output.repeat_interleave(
                    self.top_k_for_generation, dim=0
                )
                # shape: (batch_size * top_k, seq_len+1, hidden_dim)
                # +1 because we have user_id token

                repeated_encoder_attention_mask = (
                    encoder_attention_mask.repeat_interleave(
                        self.top_k_for_generation, dim=0
                    )
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
        user_id: Optional[torch.Tensor] = None,
        future_ids: Optional[torch.Tensor] = None,
        attention_mask_decoder: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder-decoder model.
        Parameters:
            attention_mask_encoder (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
            future_ids (Optional[torch.Tensor]): The future IDs for the decoder.
            attention_mask_decoder (Optional[torch.Tensor]): The attention mask for the decoder.
        """

        encoder_output, attention_mask_for_encoder = self.encoder_forward_pass(
            attention_mask=attention_mask_encoder,
            input_ids=input_ids,
            user_id=user_id,
        )

        decoder_output = self.decoder_forward_pass(
            future_ids=future_ids,
            attention_mask=attention_mask_decoder,
            encoder_output=encoder_output,
            attention_mask_for_encoder=attention_mask_for_encoder,
            use_cache=False,  # we are not using cache for training
        )
        return decoder_output

    def get_embedding_table(self, table_name: str, hierarchy: Optional[int] = None):
        """
        Get the embedding table for the given table name and hierarchy.
        Args:
            table_name: The name of the table to get the embedding for.
            hierarchy: The hierarchy level to get the embedding for.
        """
        # here we assume the encoder and decoder share the same embedding table
        # we can have flexible embedding table in the future
        if table_name == "encoder":
            embedding_table = self.item_sid_embedding_table_encoder
        elif table_name == "decoder":
            embedding_table = self.item_sid_embedding_table_encoder

        if hierarchy is not None:
            return embedding_table(
                torch.arange(
                    hierarchy * self.num_embeddings_per_hierarchy,
                    (hierarchy + 1) * self.num_embeddings_per_hierarchy,
                ).to(self.device)
            )
        return embedding_table

    def predict_step(self, batch: SequentialModelInputData):
        generated_sids, _ = self.model_step(batch)
        ids = [
            id_.item() if isinstance(id, torch.Tensor) else id
            for id_ in batch.user_id_list
        ]
        model_output = OneKeyPerPredictionOutput(
            keys=ids,
            predictions=generated_sids,
            key_name=self.prediction_key_name,
            prediction_name=self.prediction_value_name,
        )
        return model_output

    def model_step(
        self,
        model_input: SequentialModelInputData,
        label_data: Optional[SequentialModuleLabelData] = None,
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
                attention_mask=model_input.mask,
                **{
                    self.feature_to_model_input_map.get(k, k): v
                    for k, v in model_input.transformed_sequences.items()
                },
            )
            return generated_ids, 0  # returning 0 here because we don't have a loss

        fut_ids = None
        for label in label_data.labels:
            curr_label = label_data.labels[label]
            fut_ids = curr_label.reshape(model_input.mask.size(0), -1)
        # here we pass labels in to the forward function
        # because the decoder is causal and we are doing shifted prediction
        model_output = self.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=fut_ids,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        # we prepended a bos token to the decoder input
        # so we need to remove the last token in the output
        model_output = model_output[:, :-1]

        # the label locations is shared for all semantic id hierarchies
        loss = 0
        for hierarchy in range(self.num_hierarchies):
            input_ = self.decoder.decoder_mlp[hierarchy](model_output[:, hierarchy])
            loss += self.loss_function(
                input=input_,
                target=fut_ids[:, hierarchy].long(),
            )
        return model_output, loss
