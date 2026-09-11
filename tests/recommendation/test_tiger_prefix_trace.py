from __future__ import annotations

import torch
from transformers import T5Config, T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Stack

from src.data.components.data_models import TigerLabelData, TigerModelInput
from src.data.components.prefix_trace import PREFIX_TRACE_PAYLOAD_NAME, validate_prefix_trace_bundle
from src.recommendation.tiger.prefix_allocation import PrefixAllocationConfig, PrefixMassLookup
from src.recommendation.tiger.tiger import Tiger


def create_trace_tiger(
    *,
    beam_width: int = 2,
    prefix_allocation: PrefixAllocationConfig | None = None,
    item_frequencies: torch.Tensor | None = None,
) -> Tiger:
    encoder_config = T5Config(vocab_size=4, d_model=4, num_heads=2, d_ff=8, d_kv=2, num_layers=1)
    decoder_config = T5Config(
        vocab_size=4,
        d_model=4,
        num_heads=2,
        d_ff=8,
        d_kv=2,
        num_layers=1,
        is_decoder=True,
        is_encoder_decoder=False,
    )
    return Tiger(
        encoder=T5EncoderModel(encoder_config),
        decoder=T5Stack(decoder_config),
        semantic_ids=torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
        num_hierarchies=2,
        codebook_size=4,
        embedding_dim=4,
        top_k_for_generation=beam_width,
        should_check_prefix=True,
        trace_prefix_survival=True,
        prefix_trace_metadata={
            "data_split": "evaluation",
            "beam_width": beam_width,
            "num_hierarchies": 2,
            "codebook_size": 4,
            "trace_mode": "teacher_forcing_and_constrained_beam",
            "checkpoint_reference": "wandb://checkpoint",
            "semantic_id_reference": "wandb://sid",
        },
        prefix_allocation=prefix_allocation,
        item_frequencies=item_frequencies,
    ).eval()


def test_trace_enabled_generation_preserves_ids_and_scores():
    torch.manual_seed(7)
    model = create_trace_tiger()
    input_ids = torch.tensor([[0, 1], [1, 0]])
    attention_mask = torch.ones_like(input_ids)
    targets = torch.tensor([[0, 0], [1, 1]])

    with torch.no_grad():
        ordinary_ids, ordinary_scores = model.generate(attention_mask, input_ids)
        traced_ids, traced_scores, trace = model.generate(
            attention_mask,
            input_ids,
            target_ids=targets,
            trace_enabled=True,
        )

    assert torch.equal(ordinary_ids, traced_ids)
    assert torch.equal(ordinary_scores, traced_scores)
    assert trace["target_prefix_survived"].shape == (2, 2)
    assert trace["first_failure_depth"].shape == (2,)


def test_teacher_trace_uses_strict_greater_rank_for_ties_and_rejects_invalid_prefix():
    model = create_trace_tiger()
    logits = torch.zeros((1, 2, 8))
    target = torch.tensor([[0, 1]])

    trace = model.decoder.teacher_forcing_trace(logits, target)

    assert trace["teacher_legal_rank"].tolist() == [[1, 1]]
    invalid_target = torch.tensor([[3, 3]])
    try:
        model.decoder.teacher_forcing_trace(logits, invalid_target)
    except ValueError as error:
        assert "invalid catalog prefix" in str(error)
    else:
        raise AssertionError("Invalid catalog target must be rejected.")


def test_beam_observation_reconstructs_parent_and_first_failure():
    model = create_trace_tiger()
    decoder = model.decoder
    target = torch.tensor([[1, 1]])
    previous_ids = torch.tensor([[[1], [0]]])
    previous_scores = torch.tensor([[0.6, 0.4]])
    candidate_logits = torch.tensor([[0.0, 2.0, float("-inf"), float("-inf")], [2.0, 0.0, float("-inf"), float("-inf")]])
    generated_ids = torch.tensor([[[0, 0], [1, 1]]])
    scores = torch.tensor([[0.5, 0.4]])

    observation = decoder._observe_beam_step(
        candidate_logits=candidate_logits,
        previous_generated_ids=previous_ids,
        previous_scores=previous_scores,
        generated_ids=generated_ids,
        scores=scores,
        target_ids=target,
        hierarchy=1,
        batch_size=1,
    )

    assert observation["target_prefix_survived"].tolist() == [True]
    assert observation["target_beam_rank"].tolist() == [2]
    assert observation["target_parent_beam_rank"].tolist() == [1]


def test_beam_observation_uses_prefix_values_for_sorted_shortlist_membership():
    decoder = create_trace_tiger().decoder
    observation = decoder._observe_beam_step(
        candidate_logits=torch.tensor([[0.0, 2.0, 1.0, float("-inf")]]),
        previous_generated_ids=None,
        previous_scores=None,
        generated_ids=torch.tensor([[[1], [0]]]),
        scores=torch.tensor([[0.8, 0.2]]),
        target_ids=torch.tensor([[1, 1]]),
        hierarchy=0,
        batch_size=1,
        allocation_step={
            "shortlist_indices": torch.tensor([[0, 2]]),
            "shortlist_prefixes": torch.tensor([[[1], [2]]]),
            "selected_by_reserve": torch.tensor([[True, False]]),
            "reserved_count": torch.tensor([1]),
        },
    )

    assert observation["target_allocation_shortlisted"].tolist() == [True]
    assert observation["target_selected_by_reserve"].tolist() == [True]


def test_first_failure_depth_covers_first_middle_and_full_survival():
    survival = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [True, True, True],
        ]
    )

    depths = create_trace_tiger().decoder._first_failure_depth(survival)

    assert depths.tolist() == [1, 2, -1]


def test_beam_observation_uses_minus_one_when_target_parent_is_absent():
    decoder = create_trace_tiger().decoder
    observation = decoder._observe_beam_step(
        candidate_logits=torch.zeros((2, 4)),
        previous_generated_ids=torch.tensor([[[0], [0]]]),
        previous_scores=torch.tensor([[0.6, 0.4]]),
        generated_ids=torch.tensor([[[0, 0], [0, 1]]]),
        scores=torch.tensor([[0.5, 0.4]]),
        target_ids=torch.tensor([[1, 1]]),
        hierarchy=1,
        batch_size=1,
    )

    assert observation["target_prefix_survived"].tolist() == [False]
    assert observation["target_beam_rank"].tolist() == [-1]
    assert observation["target_parent_beam_rank"].tolist() == [-1]
    assert torch.isnan(observation["target_path_score"]).all()


def test_trace_predict_step_requires_labels_and_keeps_prediction_bundle_separate():
    model = create_trace_tiger()
    model_input = TigerModelInput(
        input_ids=torch.tensor([[0, 1]]),
        attention_mask=torch.tensor([[1, 1]]),
        output_keys=torch.tensor([17]),
    )
    try:
        model.predict_step((model_input, None))
    except ValueError as error:
        assert "target labels are required" in str(error)
    else:
        raise AssertionError("Trace prediction without labels must fail.")

    with torch.no_grad():
        output = model.predict_step((model_input, TigerLabelData(torch.tensor([[0, 1]]))))
    payload = output.auxiliary[PREFIX_TRACE_PAYLOAD_NAME]
    validate_prefix_trace_bundle({"keys": output.keys, **payload})
    assert output.predictions.ndim == 3
    assert set(payload) == {"schema_version", "labels", "trace", "metadata"}


def test_prefix_mass_lookup_aggregates_shared_training_support():
    lookup = PrefixMassLookup(
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
        torch.tensor([8, 2, 1, 0]),
        codebook_size=4,
        num_hierarchies=2,
    )

    assert lookup.query(torch.tensor([[0], [1]])).tolist() == [10, 1]
    assert lookup.query(torch.tensor([[0, 1], [1, 1]])).tolist() == [2, 0]
    assert lookup.summary["total_training_interactions"] == 11
    assert lookup.summary["zero_frequency_items"] == 1


def test_prefix_mass_lookup_rejects_unknown_prefix_and_misaligned_frequencies():
    semantic_ids = torch.tensor([[0, 0], [1, 1]])
    with torch.no_grad():
        try:
            PrefixMassLookup(
                semantic_ids,
                torch.tensor([1]),
                codebook_size=2,
                num_hierarchies=2,
            )
        except ValueError as error:
            assert "one value per semantic ID row" in str(error)
        else:
            raise AssertionError("Misaligned allocation frequencies must fail.")

    lookup = PrefixMassLookup(
        semantic_ids,
        torch.tensor([1, 1]),
        codebook_size=2,
        num_hierarchies=2,
    )
    try:
        lookup.query(torch.tensor([[0, 1]]))
    except ValueError as error:
        assert "missing allocation prior entries" in str(error)
    else:
        raise AssertionError("Unknown prefix must fail.")


def test_prefix_allocation_selects_low_mass_candidate_inside_shortlist_and_keeps_scores():
    model = create_trace_tiger(
        prefix_allocation=PrefixAllocationConfig(
            enabled=True,
            reserved_slots=1,
            pool_multiplier=2,
        ),
        item_frequencies=torch.tensor([100, 100, 1, 1]),
    )
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    prefixes = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]])

    selected, observation = model.decoder._select_prefix_balanced(scores, prefixes)

    assert selected.tolist() == [[0, 2]]
    torch.testing.assert_close(scores.gather(1, selected), torch.tensor([[0.9, 0.7]]))
    assert observation["selected_by_reserve"].tolist() == [[False, True]]
    assert observation["reserved_count"].tolist() == [1]


def test_prefix_allocation_excludes_catalog_invalid_shortlist_padding():
    model = create_trace_tiger(
        prefix_allocation=PrefixAllocationConfig(
            enabled=True,
            reserved_slots=1,
            pool_multiplier=2,
        ),
        item_frequencies=torch.tensor([100, 100, 1, 1]),
    )
    model.decoder.prefix_mass_lookup = PrefixMassLookup(
        torch.tensor([[0, 0], [0, 1], [1, 0]]),
        torch.tensor([100, 1, 50]),
        codebook_size=4,
        num_hierarchies=2,
    )
    scores = torch.tensor([[0.9, 0.8, 0.99, 1.0]])
    prefixes = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]])

    selected, observation = model.decoder._select_prefix_balanced(scores, prefixes)

    assert selected.tolist() == [[2, 1]]
    assert observation["selected_by_reserve"].tolist() == [[False, True]]


def test_prefix_allocation_validation_names_invalid_parameters():
    for config, message in (
        (PrefixAllocationConfig(enabled=True, reserved_slots=0), "reserved_slots"),
        (PrefixAllocationConfig(enabled=True, reserved_slots=2), "smaller than beam width"),
        (PrefixAllocationConfig(enabled=True, reserved_slots=1, pool_multiplier=0), "pool_multiplier"),
        (PrefixAllocationConfig(enabled=False, reserved_slots=1), "must be zero"),
        (PrefixAllocationConfig(source_split="testing"), "source_split"),
    ):
        try:
            create_trace_tiger(
                prefix_allocation=config,
                item_frequencies=torch.ones(4, dtype=torch.long),
            )
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"Invalid config must fail: {config}")


def test_prefix_allocation_tie_break_is_deterministic():
    model = create_trace_tiger(
        prefix_allocation=PrefixAllocationConfig(
            enabled=True,
            reserved_slots=1,
            pool_multiplier=2,
        ),
        item_frequencies=torch.ones(4, dtype=torch.long),
    )
    scores = torch.ones((1, 4))
    prefixes = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]])

    first, _ = model.decoder._select_prefix_balanced(scores, prefixes)
    second, _ = model.decoder._select_prefix_balanced(scores, prefixes)

    assert first.tolist() == second.tolist() == [[0, 1]]


def test_disabled_prefix_allocation_preserves_generation_and_checkpoint_keys():
    torch.manual_seed(11)
    baseline = create_trace_tiger()
    with_prior = create_trace_tiger(
        prefix_allocation=PrefixAllocationConfig(enabled=False, pool_multiplier=2),
        item_frequencies=torch.tensor([10, 3, 2, 0]),
    )
    with_prior.load_state_dict(baseline.state_dict())
    assert set(with_prior.state_dict()) == set(baseline.state_dict())

    input_ids = torch.tensor([[0, 1], [1, 0]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        baseline_ids, baseline_scores = baseline.generate(attention_mask, input_ids)
        prior_ids, prior_scores = with_prior.generate(attention_mask, input_ids)

    assert torch.equal(prior_ids, baseline_ids)
    assert torch.equal(prior_scores, baseline_scores)


def test_allocation_trace_records_training_only_identity_and_target_observations():
    model = create_trace_tiger(
        prefix_allocation=PrefixAllocationConfig(
            enabled=True,
            reserved_slots=1,
            pool_multiplier=2,
        ),
        item_frequencies=torch.tensor([10, 3, 2, 0]),
    )
    model_input = TigerModelInput(
        input_ids=torch.tensor([[0, 1]]),
        attention_mask=torch.tensor([[1, 1]]),
        output_keys=torch.tensor([5]),
    )

    with torch.no_grad():
        output = model.predict_step((model_input, TigerLabelData(torch.tensor([[0, 1]]))))

    payload = output.auxiliary[PREFIX_TRACE_PAYLOAD_NAME]
    validate_prefix_trace_bundle({"keys": output.keys, **payload})
    assert payload["metadata"]["prefix_allocation"] == {
        "enabled": True,
        "reserved_slots": 1,
        "pool_multiplier": 2,
        "source_split": "training",
        "strategy": "prefix_training_mass",
    }
    assert payload["metadata"]["training_frequency_summary"]["total_training_interactions"] == 15
    assert payload["trace"]["target_prefix_training_mass"].shape == (1, 2)
    assert payload["trace"]["target_allocation_shortlisted"].dtype == torch.bool
    assert payload["trace"]["target_selected_by_reserve"].dtype == torch.bool
