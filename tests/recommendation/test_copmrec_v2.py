"""v2 目标、参数梯度、禁用状态与完整恢复身份。"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from copmrec_fixtures import copmrec, snapshot
from test_liger import batch

from src.recommendation.copmrec.ablation import VARIANTS
from src.recommendation.copmrec.diagnosis import CoPMRecDiagnosis, prefix_observations


@pytest.mark.parametrize("variant", [None, *VARIANTS])
def test_each_objective_has_complete_trainable_gradients_and_restores_own_best(variant):
    original, saved = snapshot(variant)
    actual = copmrec(variant, training_model_config=None).eval()
    actual.on_load_checkpoint(saved)
    actual.load_state_dict(saved["state_dict"], strict=True)
    original.eval()
    torch.testing.assert_close(actual.retrieve(batch()[0])[0], original.retrieve(batch()[0])[0])
    assert saved["copmrec_version"] == "v2" and actual.restored_step == 1
    assert "native_view_ce" not in saved["copmrec_unified_scratch"]
    assert not hasattr(actual, "_joint_and_native_catalog_logits")
    actual._trainer = SimpleNamespace(world_size=2)
    with pytest.raises(ValueError, match="single GPU/process"):
        actual.on_predict_start()


@pytest.mark.parametrize("variant", [None, *VARIANTS])
def test_loss_is_exactly_the_declared_active_terms_and_disabled_parameters_stay_zero(variant):
    actual = copmrec(variant)
    values = actual.losses(batch()[0], batch()[1].target_ids, training=True)
    terms = ["sid_loss"]
    if variant != "no_joint_ce":
        terms.append("content_loss")
    if variant != "no_mixture":
        terms.append("mixture_loss")
    torch.testing.assert_close(values["loss"], sum(values[x] for x in terms))
    assert "native_view_ce_loss" not in values
    values["loss"].backward()
    if variant == "no_mixture":
        assert actual.dynamic_gate.bias.grad is None and actual.dynamic_gate.bias.sigmoid().item() == 0.5
    if variant == "no_residual":
        assert actual.collaborative_residual.weight.grad is None
        assert torch.count_nonzero(actual.item_content_residual(torch.arange(6))) == 0
        assert len(actual.configure_optimizers()["optimizer"].param_groups) == 1
    else:
        gradient = actual.collaborative_residual.weight.grad
        assert gradient[actual.seen_mask].abs().sum() > 0
        assert torch.count_nonzero(gradient[~actual.seen_mask]) == 0


def test_joint_ce_directly_supervises_residual():
    actual = copmrec()
    torch.nn.functional.cross_entropy(actual.dense_logits(torch.randn(2, 8)), torch.tensor([2, 3])).backward()
    assert actual.collaborative_residual.weight.grad[actual.seen_mask].abs().sum() > 0
    assert actual.content_projection.output.weight.grad.abs().sum() > 0


def test_history_and_cold_rows_remain_eligible_with_stable_ties():
    actual = copmrec().eval()
    scores = torch.zeros(2, 6)
    with patch.object(actual, "dense_logits", return_value=scores):
        predicted = actual.retrieve(batch()[0])[0]
        validation = actual.eval_step(batch())["generated_ids"]
    expected = actual.semantic_ids[:3].expand(2, -1, -1)
    torch.testing.assert_close(predicted, expected)
    torch.testing.assert_close(validation, expected)
    scores[:, 4:] = 1
    with patch.object(actual, "dense_logits", return_value=scores):
        torch.testing.assert_close(actual.retrieve(batch()[0])[0][:, :2], actual.semantic_ids[4:].expand(2, -1, -1))


@pytest.mark.parametrize("origin", [None, *VARIANTS])
@pytest.mark.parametrize("destination", [None, *VARIANTS])
def test_restore_never_silently_relabels_other_variants(origin, destination):
    if origin == destination:
        return
    _, saved = snapshot(origin)
    with pytest.raises(ValueError):
        copmrec(destination).on_load_checkpoint(saved)


@pytest.mark.parametrize(
    "fault", ["history", "native", "bool_type", "old_version", "seed", "moments", "cold", "budget", "schedule"]
)
def test_restore_rejects_invalid_contract_or_state(fault):
    _, saved = snapshot()
    if fault == "history":
        saved["copmrec_formal_release"]["inference_history_exclusion"] = True
    elif fault == "native":
        saved["copmrec_formal_release"]["native_view_ce"] = True
    elif fault == "bool_type":
        saved["copmrec_formal_release"]["native_view_ce"] = 0
    elif fault == "old_version":
        saved["copmrec_version"] = "v5.3"
    elif fault == "seed":
        saved["copmrec_formal_release"]["initialization_seed"] = 200
    elif fault == "moments":
        saved["optimizer_states"][0]["state"].clear()
    elif fault == "cold":
        saved["state_dict"]["collaborative_residual.weight"][-1, 0] = 1
    elif fault == "budget":
        saved["global_step"] = 50001
    else:
        saved["lr_schedulers"][0]["scheduler_steps"] = 100000
    with pytest.raises(ValueError):
        copmrec().on_load_checkpoint(saved)


def test_prefix_diagnostic_matches_training_mixture_nll():
    actual = copmrec().eval()
    x, y = batch()
    observed = prefix_observations(actual, x, y.target_ids)
    torch.testing.assert_close(
        torch.stack([row["mixed_nll"] for row in observed]).mean(), actual.losses(x, y.target_ids)["mixture_loss"]
    )


def test_history_exclusion_cannot_be_enabled_in_v2_diagnosis():
    with pytest.raises(ValueError):
        CoPMRecDiagnosis(backbone=copmrec(), analysis="hits", frequencies=[1, 1, 1, 1, 0, 0], exclude_history=True)


@pytest.mark.parametrize("analysis", ["hits", "residual", "prefix"])
def test_v2_mechanism_paths_use_four_current_bundles_or_own_checkpoint_and_reproduce_full(analysis):
    from src.data.components.data_models import ModelOutput

    catalog = dict(
        keys=torch.arange(12),
        semantic_ids=torch.cartesian_prod(torch.arange(3), torch.arange(4)),
        embeddings=torch.arange(60).reshape(12, 5).float() / 60,
        seen_mask=torch.tensor([True] * 8 + [False] * 4),
    )
    original, saved = snapshot(catalog=catalog, codebook_size=4, top_k=10)
    original.eval()
    x, y = batch()
    bundle = ModelOutput(x.output_keys, original.retrieve(x)[0].flip(0))
    x.output_keys = x.output_keys.flip(0)
    paths = (
        {name: bundle for name in ["full", *VARIANTS]}
        if analysis == "hits"
        else {"full": bundle}
        if analysis == "residual"
        else None
    )
    model = copmrec(catalog=catalog, codebook_size=4, top_k=10)
    diagnosis = CoPMRecDiagnosis(
        backbone=model,
        analysis=analysis,
        frequencies=[1] * 8 + [0] * 4,
        checkpoint=None if analysis == "hits" else saved,
        prediction_bundles=paths,
        bootstrap_repetitions=12,
        method_version="v2",
    )
    diagnosis._trainer = SimpleNamespace(world_size=1)
    diagnosis.on_test_start()
    diagnosis.test_step((x, y), 0)
    diagnosis.structured_analysis_output()
    assert diagnosis.metadata["ranking_history_exclusion"] is False
    assert len(diagnosis.rows) == (8 if analysis in {"hits", "residual"} else 4)
    if analysis == "residual":
        torch.testing.assert_close(torch.cat(diagnosis.predictions["v11"]), bundle.predictions.flip(0))
    if analysis == "hits":
        assert {row["variant"] for row in diagnosis.rows} == {"full", *VARIANTS}
