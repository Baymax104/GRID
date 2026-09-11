import copy

import pytest
import torch

from src.data.components.data_models import (
    DiagnosisBatch,
    PrefixTraceBundle,
    RecommendationOutcomeInput,
    SIDViews,
)
from src.quantization.tail_sid_diagnosis.candidate_allocation import (
    ACCESS_TRANSITIONS,
    TOP10_TRANSITIONS,
    build_candidate_allocation_evidence,
    cross_setting_verdict,
)
from src.quantization.tail_sid_diagnosis.evidence import build_diagnosis_evidence


def _trace(keys, labels, ranks, *, enabled, beam_width=10):
    survived = torch.tensor(
        [[True, True, rank is not None] if rank is not None else [True, False, False] for rank in ranks]
    )
    beam_rank = torch.where(survived, torch.ones_like(survived, dtype=torch.long), -1)
    for index, rank in enumerate(ranks):
        if rank is not None:
            beam_rank[index, -1] = rank
    shape = survived.shape
    return PrefixTraceBundle(
        schema_version="tiger_prefix_trace_v1",
        keys=keys.clone(),
        labels=labels.clone(),
        trace={
            "teacher_target_probability": torch.full(shape, 0.5),
            "teacher_legal_rank": torch.ones(shape, dtype=torch.long),
            "teacher_target_vs_best_legal_margin": torch.zeros(shape),
            "target_prefix_survived": survived,
            "target_beam_rank": beam_rank,
            "target_parent_beam_rank": beam_rank.clone(),
            "target_path_score": torch.full(shape, 0.2),
            "beam_cutoff_score": torch.full(shape, 0.1),
            "cutoff_margin": torch.full(shape, 0.1),
            "legal_candidate_count": torch.full(shape, 16, dtype=torch.long),
            "target_prefix_training_mass": torch.tensor(
                [[100.0, 80.0, 50.0], [90.0, 60.0, 20.0], [5.0, 2.0, 1.0], [3.0, 1.0, 0.0]]
            ),
            "target_allocation_shortlisted": torch.ones(shape, dtype=torch.bool),
            "target_selected_by_reserve": torch.full(shape, enabled, dtype=torch.bool),
            "allocation_reserved_count": torch.full(shape, 1 if enabled else 0, dtype=torch.long),
            "first_failure_depth": torch.tensor([-1 if rank is not None else 2 for rank in ranks]),
        },
        metadata={
            "data_split": "evaluation",
            "beam_width": beam_width,
            "num_hierarchies": 3,
            "codebook_size": 256,
            "trace_mode": "teacher_forcing_and_constrained_beam",
            "checkpoint_reference": "wandb://checkpoint",
            "semantic_id_reference": "wandb://sid",
            "seed": 42,
            "prefix_allocation": {
                "enabled": enabled,
                "reserved_slots": 1 if enabled else 0,
                "pool_multiplier": 2,
                "source_split": "training",
                "strategy": "prefix_training_mass",
            },
            "training_frequency_summary": {"total_training_interactions": 1000},
        },
    )


def _candidates(catalog, labels, ranks, width=10):
    rows = torch.stack([catalog[(index + 4) % len(catalog)].repeat(width, 1) for index in range(len(labels))])
    for index, rank in enumerate(ranks):
        if rank is not None:
            rows[index, rank - 1] = labels[index]
    return rows


def _fixture(*, baseline_ranks=(1, 1, None, None), intervention_ranks=(1, 1, 1, None)):
    item_ids = torch.arange(1, 13)
    catalog = torch.tensor([[index, index + 20, index + 40] for index in range(1, 13)])
    keys = torch.arange(101, 105)
    labels = catalog[:4]
    identity = {
        field: {
            "artifact_name": field,
            "artifact_version": "v1",
            "resolved_path": f"cache/{field}",
        }
        for field in {
            "semantic_id_path",
            "baseline_recommendation_output_path",
            "intervention_recommendation_output_path",
            "baseline_prefix_trace_path",
            "intervention_prefix_trace_path",
        }
    }
    batch = DiagnosisBatch(
        sid_views=SIDViews(
            item_ids=item_ids,
            raw_sid=catalog[:, :2],
            model_sid=catalog,
            dedup_digit=catalog[:, -1],
        ),
        frequencies={int(item): 13 - int(item) for item in item_ids},
        groups_by_item={},
        embeddings=None,
        recommendation=RecommendationOutcomeInput(
            user_ids=keys.clone(),
            label_item_ids=item_ids[:4],
            generated_sids=_candidates(catalog, labels, baseline_ranks),
        ),
        widened_recommendation=RecommendationOutcomeInput(
            user_ids=keys.clone(),
            label_item_ids=item_ids[:4],
            generated_sids=_candidates(catalog, labels, intervention_ranks),
        ),
        fixed_prefix_trace=_trace(keys, labels, baseline_ranks, enabled=False),
        widened_prefix_trace=_trace(keys, labels, intervention_ranks, enabled=True),
        input_metadata={
            "baseline_recommendation_output_reference": "wandb://baseline",
            "intervention_recommendation_output_reference": "wandb://intervention",
            "baseline_prefix_trace_reference": "wandb://baseline",
            "intervention_prefix_trace_reference": "wandb://intervention",
            "resolved_artifact_identity": identity,
        },
    )
    item_rows = [
        {
            "item_id": int(item),
            "group": "Head" if index < 2 else ("Tail" if index < 4 else "Mid"),
            "freq_train": 13 - int(item),
            "raw_damage": 0.0,
        }
        for index, item in enumerate(item_ids)
    ]
    return batch, item_rows


def _build(batch, item_rows):
    return build_candidate_allocation_evidence(batch, item_rows, enabled=True)


def test_probe_pair_reports_access_top10_and_layer_mechanism_separately():
    batch, item_rows = _fixture()

    evidence = _build(batch, item_rows)

    assert evidence.summary["identity"]["identity_verified"] is True
    users = evidence.tables["candidate_allocation_by_user.csv"]
    assert sum(row["top10_transition"] == "miss_to_hit" for row in users) == 1
    tail = next(
        row for row in evidence.tables["candidate_allocation_by_group.csv"]
        if row["group"] == "Tail+Tail-Cold"
    )
    assert tail["candidate_added"] == 1
    assert tail["top10_added"] == 1
    assert tail["candidate_access_delta"] == pytest.approx(0.5)
    assert sum(tail[name] for name in TOP10_TRANSITIONS) == tail["support"]
    assert sum(tail[name] for name in ACCESS_TRANSITIONS) == tail["support"]
    layers = evidence.tables["candidate_allocation_by_layer.csv"]
    final_tail = next(
        row for row in layers if row["group"] == "Tail+Tail-Cold" and row["layer"] == 3
    )
    assert final_tail["survival_delta"] == pytest.approx(0.5)
    assert final_tail["allocation_reserved_count_mean"] == pytest.approx(1.0)
    assert evidence.summary["setting_gate"]["setting_gate_passed"] is True
    assert evidence.summary["cross_setting_verdict"]["verdict"] == "inconclusive"


def test_full_diagnosis_routes_equal_width_pair_to_candidate_allocation_probe():
    batch, item_rows = _fixture()
    batch.groups_by_item = {int(row["item_id"]): str(row["group"]) for row in item_rows}

    evidence = build_diagnosis_evidence(
        batch,
        bootstrap_samples=10,
        semantic_reference_pairs=10,
        candidate_allocation_probe_enabled=True,
    )

    assert evidence.summary["prefix_mechanism"]["available"] is True
    assert evidence.summary["prefix_mechanism"]["recovery_available"] is False
    assert evidence.summary["candidate_allocation"]["available"] is True
    assert "prefix_widened_recovery.csv" not in evidence.to_structured_output().tables
    assert "candidate_allocation_by_user.csv" in evidence.to_structured_output().tables


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("keys", "user keys"),
        ("labels", "labels"),
        ("split", "data_split mismatch"),
        ("checkpoint", "checkpoint_reference mismatch"),
        ("sid", "semantic_id_reference mismatch"),
        ("seed", "seed mismatch"),
        ("width", "beam_width mismatch"),
        ("baseline_enabled", "Baseline Prefix Trace must"),
        ("intervention_disabled", "Intervention Prefix Trace must"),
        ("candidate_catalog", "absent from the model catalog"),
        ("lineage", "lineage conflict"),
        ("artifact_identity", "identity is unverified"),
    ],
)
def test_probe_pair_audit_rejects_identity_mismatch(mutation, message):
    batch, item_rows = _fixture()
    if mutation == "keys":
        batch.widened_recommendation.user_ids[0] = 999
    elif mutation == "labels":
        batch.widened_prefix_trace.labels[0] = batch.widened_prefix_trace.labels[1]
    elif mutation in {"split", "checkpoint", "sid", "seed"}:
        field = {
            "split": "data_split",
            "checkpoint": "checkpoint_reference",
            "sid": "semantic_id_reference",
            "seed": "seed",
        }[mutation]
        batch.widened_prefix_trace.metadata[field] = "other"
    elif mutation == "width":
        batch.widened_prefix_trace.metadata["beam_width"] = 20
    elif mutation == "baseline_enabled":
        batch.fixed_prefix_trace.metadata["prefix_allocation"]["enabled"] = True
    elif mutation == "intervention_disabled":
        batch.widened_prefix_trace.metadata["prefix_allocation"]["enabled"] = False
    elif mutation == "candidate_catalog":
        batch.widened_recommendation.generated_sids[0, 2] = torch.tensor([99, 98, 97])
    elif mutation == "lineage":
        batch.input_metadata["intervention_prefix_trace_reference"] = "wandb://other"
    else:
        del batch.input_metadata["resolved_artifact_identity"]["intervention_prefix_trace_path"]

    with pytest.raises((ValueError, KeyError), match=message):
        _build(batch, item_rows)


def test_probe_gate_rejects_intermediate_only_improvement_and_head_cost():
    batch, item_rows = _fixture(
        baseline_ranks=(1, 1, None, None),
        intervention_ranks=(1, 1, None, None),
    )
    batch.widened_prefix_trace.trace["target_prefix_survived"][2, 1] = True
    evidence = _build(batch, item_rows)
    assert evidence.summary["setting_gate"]["setting_gate_passed"] is False
    assert evidence.summary["setting_gate"]["tail_top10_added"] == 0

    costly_batch, costly_items = _fixture(
        baseline_ranks=(1, 1, None, None),
        intervention_ranks=(1, None, 1, None),
    )
    costly = _build(costly_batch, costly_items)
    assert costly.summary["setting_gate"]["head_guardrail_passed"] is False
    assert costly.summary["setting_gate"]["setting_gate_passed"] is False


def test_cross_setting_verdict_requires_four_settings_and_all_gates():
    passing = {
        "tail_candidate_access_improved": True,
        "tail_top10_added": 1,
        "overall_guardrail_passed": True,
        "head_guardrail_passed": True,
    }
    no_access = {**passing, "tail_candidate_access_improved": False, "tail_top10_added": 0}

    assert cross_setting_verdict([passing] * 3)["verdict"] == "inconclusive"
    assert cross_setting_verdict([passing, passing, passing, no_access])["verdict"] == "advance"
    no_addition = {**passing, "tail_top10_added": 0}
    assert cross_setting_verdict([passing, no_addition, no_addition, no_access])["verdict"] == "stop"


def test_probe_pair_rejects_rank_membership_and_missing_allocation_trace_fields():
    batch, item_rows = _fixture()
    rank_mismatch = copy.deepcopy(batch)
    rank_mismatch.widened_prefix_trace.trace["target_beam_rank"][2, -1] = 2
    with pytest.raises(ValueError, match="target rank"):
        _build(rank_mismatch, item_rows)

    missing_field = copy.deepcopy(batch)
    del missing_field.widened_prefix_trace.trace["target_selected_by_reserve"]
    with pytest.raises(ValueError, match="missing allocation tensors"):
        _build(missing_field, item_rows)
