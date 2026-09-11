import copy

import pytest
import torch

from src.data.components.data_models import (
    DiagnosisBatch,
    PrefixTraceBundle,
    RecommendationOutcomeInput,
    SIDViews,
)
from src.quantization.tail_sid_diagnosis.search_ranking import (
    TRANSITIONS,
    _aggregate_items,
    _bootstrap_intervals,
    _overlap_estimate,
    build_search_ranking_evidence,
)


def _trace(keys: torch.Tensor, labels: torch.Tensor, ranks: list[int | None], beam_width: int):
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
        },
    )


def _candidates(labels: torch.Tensor, ranks: list[int | None], width: int, filler: torch.Tensor):
    rows = filler.repeat(len(labels), width, 1)
    for index, rank in enumerate(ranks):
        if rank is not None:
            rows[index, rank - 1] = labels[index]
    return rows


def _paired_fixture():
    item_ids = torch.arange(1, 9)
    catalog = torch.tensor([[index, index + 20, index + 40] for index in range(1, 9)])
    keys = torch.arange(101, 107)
    labels = catalog[:6]
    fixed_ranks = [1, 1, 1, None, None, None]
    widened_ranks = [1, 11, None, 1, 11, None]
    fixed = RecommendationOutcomeInput(
        user_ids=keys.clone(),
        label_item_ids=item_ids[:6],
        generated_sids=_candidates(labels, fixed_ranks, 10, catalog[-1]),
    )
    widened = RecommendationOutcomeInput(
        user_ids=keys.clone(),
        label_item_ids=item_ids[:6],
        generated_sids=_candidates(labels, widened_ranks, 12, catalog[-1]),
    )
    identity = {
        field: {
            "artifact_name": field,
            "artifact_version": "v1",
            "resolved_path": f"cache/{field}",
        }
        for field in {
            "semantic_id_path",
            "recommendation_output_path",
            "widened_recommendation_output_path",
            "fixed_prefix_trace_path",
            "widened_prefix_trace_path",
        }
    }
    batch = DiagnosisBatch(
        sid_views=SIDViews(
            item_ids=item_ids,
            raw_sid=catalog[:, :2],
            model_sid=catalog,
            dedup_digit=catalog[:, -1],
        ),
        frequencies={int(item): 9 - int(item) for item in item_ids},
        groups_by_item={},
        embeddings=None,
        recommendation=fixed,
        widened_recommendation=widened,
        fixed_prefix_trace=_trace(keys, labels, fixed_ranks, 10),
        widened_prefix_trace=_trace(keys, labels, widened_ranks, 50),
        input_metadata={
            "recommendation_output_reference": "wandb://fixed",
            "widened_recommendation_output_reference": "wandb://widened",
            "fixed_prefix_trace_path": "wandb://fixed",
            "widened_prefix_trace_path": "wandb://widened",
            "resolved_artifact_identity": identity,
        },
    )
    item_rows = [
        {
            "item_id": int(item),
            "group": ("Head", "Mid", "Tail", "Tail-Cold")[min(index // 2, 3)],
            "freq_train": 9 - int(item),
            "raw_damage": float(index % 2),
        }
        for index, item in enumerate(item_ids)
    ]
    return batch, item_rows


def _build(batch, item_rows):
    return build_search_ranking_evidence(
        batch,
        item_rows,
        search_ranking_enabled=True,
        risk_standardization_enabled=False,
        bin_count=5,
        bin_count_sensitivity=(3, 5, 10),
        min_items_per_group_per_bin=1,
        min_item_retention=0.5,
        max_abs_raw_damage_smd=0.1,
        bootstrap_samples=20,
        bootstrap_confidence=0.95,
        min_bootstrap_valid_fraction=0.5,
        seed=42,
    )


def test_six_state_decomposition_and_group_conservation():
    batch, item_rows = _paired_fixture()
    evidence = _build(batch, item_rows)

    users = evidence.tables["search_ranking_by_user.csv"]
    assert [row["transition"] for row in users] == list(TRANSITIONS)
    assert evidence.summary["identity"]["identity_verified"] is True
    all_row = next(row for row in evidence.tables["search_ranking_by_group.csv"] if row["group"] == "All")
    assert sum(all_row[name] for name in TRANSITIONS) == all_row["support"] == 6
    assert all_row["fixed_hit10"] == pytest.approx(0.5)
    assert all_row["widened_hit10"] == pytest.approx(2 / 6)
    assert all_row["oracle_hit10_ceiling"] == pytest.approx(4 / 6)
    assert all_row["oracle_hit10_ceiling"] >= all_row["widened_hit10"]
    assert all_row["top10_net_change"] == -1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("keys", "user keys"),
        ("label", "labels"),
        ("illegal_sid", "absent from the model catalog"),
        ("rank", "target rank"),
        ("lineage", "lineage conflict"),
    ],
)
def test_pair_audit_rejects_mismatched_inputs(mutation, message):
    batch, item_rows = _paired_fixture()
    if mutation == "keys":
        batch.widened_recommendation.user_ids[0] = 999
    elif mutation == "label":
        batch.widened_prefix_trace.labels[0] = batch.widened_prefix_trace.labels[1]
    elif mutation == "illegal_sid":
        batch.widened_recommendation.generated_sids[0, 2] = torch.tensor([99, 99, 99])
    elif mutation == "rank":
        batch.widened_prefix_trace.trace["target_beam_rank"][0, -1] = 2
    else:
        batch.input_metadata["fixed_prefix_trace_path"] = "wandb://other"
    with pytest.raises((ValueError, KeyError), match=message):
        _build(batch, item_rows)


def _item_users_and_trace():
    users = []
    survival_rows = []
    for index, (group, damage, survival) in enumerate(
        [
            ("Head", 0.0, [1, 1]),
            ("Head", 1.0, [1, 0]),
            ("Tail", 0.0, [1, 0]),
            ("Tail", 1.0, [0, 0]),
        ]
    ):
        for repeat in range(index + 1):
            users.append(
                {
                    "user_id": 100 * index + repeat,
                    "item_id": index + 1,
                    "group": group,
                    "frequency": 10,
                    "raw_damage": damage,
                    "model_sid": f"{index // 2} {index} 0",
                    "trace_index": len(survival_rows),
                }
            )
            survival_rows.append(survival)
    trace = _trace(
        torch.arange(len(users)),
        torch.zeros((len(users), 3), dtype=torch.long),
        [1] * len(users),
        10,
    )
    trace.labels = torch.zeros((len(users), 2), dtype=torch.long)
    trace.trace["target_prefix_survived"] = torch.tensor(survival_rows, dtype=torch.bool)
    return users, trace


def test_item_macro_aggregation_is_order_invariant_and_not_user_weighted():
    users, trace = _item_users_and_trace()
    forward = _aggregate_items(users, trace)
    reverse = _aggregate_items(list(reversed(users)), trace)

    assert [(row["item_id"], row["survival"].tolist()) for row in forward] == [
        (row["item_id"], row["survival"].tolist()) for row in reverse
    ]
    tail_item_macro = sum(float(row["survival"][0]) for row in forward if row["group"] == "Tail") / 2
    tail_user_macro = sum(row_values[0] for row_values in trace.trace["target_prefix_survived"][3:].tolist()) / 7
    assert tail_item_macro == pytest.approx(0.5)
    assert tail_user_macro != pytest.approx(tail_item_macro)


def test_overlap_estimator_and_cluster_bootstrap_are_reproducible():
    users, trace = _item_users_and_trace()
    items = _aggregate_items(users, trace)
    edges = [0.0, 0.5, 1.0]
    overlap, layers, metadata = _overlap_estimate(
        items, edges=edges, min_support=1, min_retention=0.5, max_abs_smd=0.1
    )

    assert metadata["available"] is True
    assert metadata["quality_status"] == "qualified"
    assert len(overlap) == 2
    assert layers[0]["standardized_survival_difference"] == pytest.approx(-0.5)
    first = _bootstrap_intervals(
        items,
        edges=edges,
        min_support=1,
        min_retention=0.5,
        max_abs_smd=0.1,
        bootstrap_samples=30,
        confidence=0.9,
        min_valid_fraction=0.1,
        seed=42,
    )
    second = _bootstrap_intervals(
        copy.deepcopy(items),
        edges=edges,
        min_support=1,
        min_retention=0.5,
        max_abs_smd=0.1,
        bootstrap_samples=30,
        confidence=0.9,
        min_valid_fraction=0.1,
        seed=42,
    )
    assert first == second
    assert all(row["interval_scope"] == "pointwise_exploratory" for row in first)
    assert all("standardized_survival_ci_lower" in row for row in first)
    assert all("raw_survival_ci_lower" in row for row in first)
    assert all("standardized_conditional_failure_ci_lower" in row for row in first)


def test_overlap_estimator_reports_no_overlap_and_zero_denominators():
    users, trace = _item_users_and_trace()
    items = _aggregate_items(users, trace)
    _, layers, metadata = _overlap_estimate(
        items, edges=[0.0, 0.5, 1.0], min_support=2, min_retention=0.8, max_abs_smd=0.1
    )
    assert layers == []
    assert metadata["available"] is False
    assert metadata["quality_status"] == "insufficient_overlap"


def test_dataset_requires_all_paired_inputs_when_search_analysis_is_enabled():
    from types import SimpleNamespace

    from src.data.datasets import DiagnosisDataset

    dataset = DiagnosisDataset(
        dataset_config=SimpleNamespace(preprocessing_functions=[]),
        data_folder="data/beauty",
        semantic_id_path="semantic.pt",
        raw_num_hierarchies=3,
        search_ranking_enabled=True,
    )
    with pytest.raises(ValueError, match="missing=.*widened_recommendation_output_path"):
        dataset._build_batch()
