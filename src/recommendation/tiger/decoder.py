import torch
import torch.nn.functional as F
import transformers
from torch import nn

from src.recommendation.tiger.prefix_allocation import PrefixAllocationConfig, PrefixMassLookup
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
        prefix_allocation: PrefixAllocationConfig | None = None,
        item_frequencies: torch.Tensor | None = None,
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
        self.prefix_allocation = prefix_allocation or PrefixAllocationConfig()
        self.prefix_allocation.validate(self.top_k_for_generation)
        self.prefix_mass_lookup = None
        if item_frequencies is not None:
            self.prefix_mass_lookup = PrefixMassLookup(
                self.semantic_ids,
                item_frequencies,
                codebook_size=self.codebook_size,
                num_hierarchies=self.num_hierarchies,
            )
        if self.prefix_allocation.enabled and self.prefix_mass_lookup is None:
            raise ValueError("item_frequencies are required when prefix allocation is enabled.")
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

    def _valid_next_token_mask(self, target_ids: torch.Tensor, hierarchy: int) -> torch.Tensor:
        """Return catalog-valid continuations for each ground-truth parent prefix."""
        batch_size = target_ids.size(0)
        candidates = torch.arange(self.codebook_size, device=target_ids.device)
        if hierarchy == 0:
            candidate_prefixes = candidates.unsqueeze(1).repeat(batch_size, 1)
        else:
            parents = target_ids[:, :hierarchy].repeat_interleave(self.codebook_size, dim=0)
            candidate_prefixes = torch.cat(
                [parents, candidates.repeat(batch_size).unsqueeze(1)],
                dim=1,
            )
        return self._check_valid_prefix(candidate_prefixes).reshape(batch_size, self.codebook_size)

    def teacher_forcing_trace(
        self,
        global_logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Derive target statistics from existing teacher-forcing logits."""
        expected_shape = (target_ids.size(0), self.num_hierarchies)
        if tuple(target_ids.shape) != expected_shape:
            raise ValueError(
                f"Trace target_ids must have shape {expected_shape}, got {tuple(target_ids.shape)}."
            )
        if global_logits.ndim != 3 or global_logits.size(1) != self.num_hierarchies:
            raise ValueError("Teacher-forcing logits have an incompatible hierarchy dimension.")
        if torch.any((target_ids < 0) | (target_ids >= self.codebook_size)):
            raise ValueError("Trace target_ids contain a token outside the hierarchy-local vocabulary.")

        probabilities = []
        legal_ranks = []
        margins = []
        for hierarchy in range(self.num_hierarchies):
            start = hierarchy * self.codebook_size
            local_logits = global_logits[:, hierarchy, start : start + self.codebook_size]
            target_token = target_ids[:, hierarchy].long()
            valid_mask = self._valid_next_token_mask(target_ids, hierarchy)
            target_is_valid = valid_mask.gather(1, target_token.unsqueeze(1)).squeeze(1)
            if not bool(target_is_valid.all()):
                invalid_rows = (~target_is_valid).nonzero(as_tuple=False).reshape(-1).tolist()
                raise ValueError(f"Trace target_ids contain an invalid catalog prefix at rows {invalid_rows[:5]}.")

            target_logits = local_logits.gather(1, target_token.unsqueeze(1)).squeeze(1)
            legal_logits = local_logits.masked_fill(~valid_mask, float("-inf"))
            probabilities.append(
                F.softmax(local_logits, dim=-1).gather(1, target_token.unsqueeze(1)).squeeze(1)
            )
            legal_ranks.append(
                ((legal_logits > target_logits.unsqueeze(1)) & valid_mask).sum(dim=1).long() + 1
            )
            margins.append(target_logits - legal_logits.max(dim=1).values)

        return {
            "teacher_target_probability": torch.stack(probabilities, dim=1),
            "teacher_legal_rank": torch.stack(legal_ranks, dim=1),
            "teacher_target_vs_best_legal_margin": torch.stack(margins, dim=1),
        }

    def _observe_beam_step(
        self,
        *,
        candidate_logits: torch.Tensor,
        previous_generated_ids: torch.Tensor | None,
        previous_scores: torch.Tensor | None,
        generated_ids: torch.Tensor,
        scores: torch.Tensor,
        target_ids: torch.Tensor,
        hierarchy: int,
        batch_size: int,
        allocation_step: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Observe one constrained beam step without changing its decisions."""
        target_token = target_ids[:, hierarchy].long()
        valid_mask = self._valid_next_token_mask(target_ids, hierarchy)
        legal_count = valid_mask.sum(dim=1).long()
        nan = torch.full((batch_size,), float("nan"), device=scores.device, dtype=scores.dtype)

        if hierarchy == 0:
            parent_rank = torch.ones(batch_size, dtype=torch.long, device=scores.device)
            local_probabilities = F.softmax(candidate_logits, dim=-1)
            target_score = local_probabilities.gather(1, target_token.unsqueeze(1)).squeeze(1)
        else:
            assert previous_generated_ids is not None and previous_scores is not None
            parents = previous_generated_ids.reshape(batch_size, self.top_k_for_generation, hierarchy)
            parent_matches = torch.all(parents == target_ids[:, None, :hierarchy], dim=2)
            parent_exists = parent_matches.any(dim=1)
            parent_index = parent_matches.long().argmax(dim=1)
            parent_rank = torch.where(
                parent_exists,
                parent_index + 1,
                torch.full_like(parent_index, -1),
            )

            local_probabilities = F.softmax(
                candidate_logits.reshape(batch_size, self.top_k_for_generation, self.codebook_size),
                dim=-1,
            )
            batch_index = torch.arange(batch_size, device=scores.device)
            continuation = local_probabilities[batch_index, parent_index, target_token]
            parent_score = previous_scores[batch_index, parent_index]
            reachable_score = parent_score * continuation
            target_score = torch.where(parent_exists, reachable_score, nan)

        target_matches = torch.all(
            generated_ids == target_ids[:, None, : hierarchy + 1],
            dim=2,
        )
        survived = target_matches.any(dim=1)
        target_index = target_matches.long().argmax(dim=1)
        beam_rank = torch.where(
            survived,
            target_index + 1,
            torch.full_like(target_index, -1),
        )
        cutoff_score = scores[:, -1]
        target_prefix_mass = nan.clone()
        target_shortlisted = torch.zeros(batch_size, dtype=torch.bool, device=scores.device)
        target_selected_by_reserve = torch.zeros(batch_size, dtype=torch.bool, device=scores.device)
        allocation_reserved_count = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
        if self.prefix_mass_lookup is not None:
            target_prefix_mass = self.prefix_mass_lookup.query(
                target_ids[:, : hierarchy + 1]
            ).to(scores.dtype)
        if allocation_step is not None:
            shortlist_prefixes = allocation_step["shortlist_prefixes"]
            target_shortlisted = torch.all(
                shortlist_prefixes == target_ids[:, None, : hierarchy + 1],
                dim=2,
            ).any(dim=1)
            selected_by_reserve = allocation_step["selected_by_reserve"]
            target_selected_by_reserve = survived & selected_by_reserve.gather(
                1, target_index.unsqueeze(1)
            ).squeeze(1)
            allocation_reserved_count = allocation_step["reserved_count"]
        return {
            "target_prefix_survived": survived,
            "target_beam_rank": beam_rank.long(),
            "target_parent_beam_rank": parent_rank.long(),
            "target_path_score": target_score,
            "beam_cutoff_score": cutoff_score,
            "cutoff_margin": target_score - cutoff_score,
            "legal_candidate_count": legal_count,
            "target_prefix_training_mass": target_prefix_mass,
            "target_allocation_shortlisted": target_shortlisted,
            "target_selected_by_reserve": target_selected_by_reserve,
            "allocation_reserved_count": allocation_reserved_count,
        }

    @staticmethod
    def _first_failure_depth(survival: torch.Tensor) -> torch.Tensor:
        """Return the first 1-based failed hierarchy, or -1 for full survival."""
        if survival.ndim != 2 or survival.dtype != torch.bool:
            raise ValueError("Prefix survival must be a 2-D bool tensor.")
        failures = ~survival
        first_failure = failures.long().argmax(dim=1) + 1
        return torch.where(
            failures.any(dim=1),
            first_failure,
            torch.full_like(first_failure, -1),
        )

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
            flattened_scores = proba
            flattened_tokens = indices
            candidate_prefixes = indices.unsqueeze(-1)
            if self.prefix_allocation.enabled:
                selected_positions, allocation_step = self._select_prefix_balanced(
                    flattened_scores,
                    candidate_prefixes,
                )
                proba_topk = flattened_scores.gather(1, selected_positions)
                indices_topk = flattened_tokens.gather(1, selected_positions)
            else:
                proba_topk, indices_topk = (
                    proba[:, : self.top_k_for_generation],
                    indices[:, : self.top_k_for_generation],
                )
                allocation_step = self._allocation_observation(
                    flattened_scores,
                    candidate_prefixes,
                    torch.arange(
                        self.top_k_for_generation,
                        device=proba.device,
                    ).unsqueeze(0).expand(batch_size, -1),
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
            parents = generated_ids.reshape(batch_size, self.top_k_for_generation, hierarchy)
            candidate_prefixes = torch.cat(
                [
                    parents.unsqueeze(2).expand(-1, -1, self.codebook_size, -1),
                    indices.reshape(batch_size, self.top_k_for_generation, self.codebook_size).unsqueeze(-1),
                ],
                dim=-1,
            ).reshape(batch_size, -1, hierarchy + 1)
            safe_proba = torch.nan_to_num(proba, nan=-1)
            if self.prefix_allocation.enabled:
                selected_positions, allocation_step = self._select_prefix_balanced(
                    safe_proba,
                    candidate_prefixes,
                )
                proba_topk = safe_proba.gather(1, selected_positions)
                indices_topk = selected_positions
            else:
                topk_results = torch.topk(safe_proba, k=self.top_k_for_generation, dim=-1)
                proba_topk, indices_topk = topk_results.values, topk_results.indices
                allocation_step = self._allocation_observation(
                    safe_proba,
                    candidate_prefixes,
                    indices_topk,
                )
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

        return generated_ids, proba_topk, allocation_step

    def _allocation_observation(
        self,
        scores: torch.Tensor,
        candidate_prefixes: torch.Tensor,
        selected_positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eligible_scores = scores
        if self.prefix_mass_lookup is not None:
            _, found = self.prefix_mass_lookup.query_with_found(
                candidate_prefixes.reshape(-1, candidate_prefixes.size(-1))
            )
            eligible_scores = scores.masked_fill(~found.reshape_as(scores), float("-inf"))
        pool_size = min(
            scores.size(1),
            self.top_k_for_generation * self.prefix_allocation.pool_multiplier,
        )
        shortlist_indices = torch.argsort(
            eligible_scores,
            dim=1,
            descending=True,
            stable=True,
        )[:, :pool_size]
        return {
            "shortlist_indices": shortlist_indices,
            "shortlist_prefixes": candidate_prefixes.gather(
                1,
                shortlist_indices.unsqueeze(-1).expand(-1, -1, candidate_prefixes.size(-1)),
            ),
            "selected_by_reserve": torch.zeros_like(selected_positions, dtype=torch.bool),
            "reserved_count": torch.zeros(scores.size(0), dtype=torch.long, device=scores.device),
        }

    def _select_prefix_balanced(
        self,
        scores: torch.Tensor,
        candidate_prefixes: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.prefix_mass_lookup is None:
            raise RuntimeError("Prefix mass lookup is required for allocation selection.")
        all_masses, found = self.prefix_mass_lookup.query_with_found(
            candidate_prefixes.reshape(-1, candidate_prefixes.size(-1))
        )
        found = found.reshape_as(scores)
        if torch.any(found.sum(dim=1) < self.top_k_for_generation):
            raise ValueError("Fewer legal catalog candidates than the configured beam width.")
        eligible_scores = scores.masked_fill(~found, float("-inf"))
        pool_size = min(
            scores.size(1),
            self.top_k_for_generation * self.prefix_allocation.pool_multiplier,
        )
        shortlist_indices = torch.argsort(
            eligible_scores,
            dim=1,
            descending=True,
            stable=True,
        )[:, :pool_size]
        shortlist_found = found.gather(1, shortlist_indices)
        prefix_masses = all_masses.reshape_as(scores).gather(1, shortlist_indices).masked_fill(
            ~shortlist_found,
            torch.iinfo(all_masses.dtype).max,
        )
        rarity_order = torch.argsort(prefix_masses, dim=1, descending=False, stable=True)
        reserve_count = min(self.prefix_allocation.reserved_slots, pool_size)
        reserve_in_shortlist = rarity_order[:, :reserve_count]
        reserved_positions = shortlist_indices.gather(1, reserve_in_shortlist)

        is_reserved = torch.zeros_like(shortlist_indices, dtype=torch.bool)
        is_reserved.scatter_(1, reserve_in_shortlist, True)
        shortlist_scores = eligible_scores.gather(1, shortlist_indices)
        backfill_order = torch.argsort(
            shortlist_scores.masked_fill(is_reserved, float("-inf")),
            dim=1,
            descending=True,
            stable=True,
        )[:, : self.top_k_for_generation - reserve_count]
        backfill_positions = shortlist_indices.gather(1, backfill_order)
        selected_positions = torch.cat([reserved_positions, backfill_positions], dim=1)
        selected_scores = scores.gather(1, selected_positions)
        final_order = torch.argsort(selected_scores, dim=1, descending=True, stable=True)
        selected_positions = selected_positions.gather(1, final_order)
        if not bool(found.gather(1, selected_positions).all()):
            raise RuntimeError("Prefix allocation selected a candidate outside the semantic-ID catalog.")

        selected_by_reserve = torch.cat(
            [
                torch.ones_like(reserved_positions, dtype=torch.bool),
                torch.zeros_like(backfill_positions, dtype=torch.bool),
            ],
            dim=1,
        ).gather(1, final_order)
        return selected_positions, {
            "shortlist_indices": shortlist_indices,
            "shortlist_prefixes": candidate_prefixes.gather(
                1,
                shortlist_indices.unsqueeze(-1).expand(-1, -1, candidate_prefixes.size(-1)),
            ),
            "selected_by_reserve": selected_by_reserve,
            "reserved_count": torch.full(
                (scores.size(0),),
                reserve_count,
                dtype=torch.long,
                device=scores.device,
            ),
        }

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
        target_ids: torch.Tensor | None = None,
        trace_enabled: bool = False,
    ):
        """Generate semantic IDs using autoregressive beam search."""
        if trace_enabled:
            if target_ids is None:
                raise ValueError("target_ids are required for prefix survival tracing.")
            if tuple(target_ids.shape) != (batch_size, self.num_hierarchies):
                raise ValueError(
                    "Trace target_ids must have shape "
                    f"({batch_size}, {self.num_hierarchies}), got {tuple(target_ids.shape)}."
                )
            target_ids = target_ids.to(encoder_output.device).long()

        generated_ids = None
        marginal_log_prob = None
        trace_steps: list[dict[str, torch.Tensor]] = []

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

            previous_generated_ids = generated_ids
            previous_scores = marginal_log_prob
            generated_ids, marginal_log_prob, allocation_step = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                marginal_log_prob=marginal_log_prob,
                hierarchy=hierarchy,
                batch_size=batch_size,
            )

            if trace_enabled:
                trace_steps.append(
                    self._observe_beam_step(
                        candidate_logits=candidate_logits,
                        previous_generated_ids=previous_generated_ids,
                        previous_scores=previous_scores,
                        generated_ids=generated_ids,
                        scores=marginal_log_prob,
                        target_ids=target_ids,
                        hierarchy=hierarchy,
                        batch_size=batch_size,
                        allocation_step=allocation_step,
                    )
                )

        if not trace_enabled:
            return generated_ids, marginal_log_prob

        beam_trace = {
            field_name: torch.stack([step[field_name] for step in trace_steps], dim=1)
            for field_name in trace_steps[0]
        }
        beam_trace["first_failure_depth"] = self._first_failure_depth(
            beam_trace["target_prefix_survived"]
        )
        return generated_ids, marginal_log_prob, beam_trace
