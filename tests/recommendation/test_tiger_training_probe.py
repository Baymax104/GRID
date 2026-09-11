from functools import partial

import pytest
import torch
from transformers import T5Config, T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Stack

from src.common.configs.model import TrainingModelConfig
from src.data.components.data_models import TigerLabelData, TigerModelInput
from src.recommendation.tiger.tiger import Tiger
from src.recommendation.tiger_training_probe.module import TigerTrainingProbe
from src.recommendation.tiger_training_probe.objective import ConditionalBranchObjective

SIDS = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])


def model_kwargs():
    config = dict(vocab_size=4, d_model=4, num_heads=2, d_ff=8, d_kv=2, num_layers=1, dropout_rate=0.0)
    return dict(
        encoder=T5EncoderModel(T5Config(**config)),
        decoder=T5Stack(T5Config(**config, is_decoder=True, is_encoder_decoder=False)),
        semantic_ids=SIDS.clone(),
        num_hierarchies=2,
        codebook_size=4,
        embedding_dim=4,
        top_k_for_generation=2,
        should_check_prefix=True,
        training_model_config=TrainingModelConfig(
            loss_function=torch.nn.CrossEntropyLoss(ignore_index=-1), optimizer=partial(torch.optim.Adam, lr=5e-5)
        ),
    )


def make_pair(tmp_path, arm="ce", alpha=0.25):
    base = Tiger(**model_kwargs())
    path = tmp_path / "initial.ckpt"
    torch.save({"state_dict": base.state_dict(), "global_step": 19900, "optimizer_states": [{"ignored": True}]}, path)
    probe = TigerTrainingProbe(
        **model_kwargs(),
        initialization_checkpoint_path=str(path),
        arm=arm,
        layers=[2],
        alpha=alpha,
        statistics={
            "semantic_ids": SIDS,
            "expected_counts": torch.tensor([9.0, 1.0, 1.0, 9.0]),
            "metadata": {"source_split": "training"},
        },
    )
    return base, probe


def batch():
    inputs = torch.tensor([[0, 1], [1, 0]])
    return TigerModelInput(input_ids=inputs, attention_mask=torch.ones_like(inputs)), TigerLabelData(
        target_ids=SIDS[[1, 2]]
    )


@pytest.mark.parametrize("arm,alpha", [("ce", 0.25), ("reweighted", 0.0)])
def test_neutral_probe_loss_gradients_update_and_checkpoint_compatible(tmp_path, arm, alpha):
    base, probe = make_pair(tmp_path, arm, alpha)
    assert set(base.state_dict()) == set(probe.state_dict())
    optimizers = [torch.optim.Adam(m.parameters(), lr=5e-5) for m in (base, probe)]
    losses = []
    for model in (base, probe):
        losses.append(model.training_step(batch(), 0)["loss"])
        losses[-1].backward()
    torch.testing.assert_close(*losses)
    for a, b in zip(base.parameters(), probe.parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)
    for optimizer in optimizers:
        assert not optimizer.state
        optimizer.step()
    for a, b in zip(base.parameters(), probe.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    base.load_state_dict(probe.state_dict(), strict=True)
    checkpoint = {}
    probe.on_save_checkpoint(checkpoint)
    assert checkpoint["training_frequency_probe"]["optimizer_initialization"] == "fresh"
    assert "lookup" in checkpoint["training_frequency_probe"]


def test_intervention_changes_training_but_preserves_evaluation_and_generation(tmp_path):
    base, probe = make_pair(tmp_path, "reweighted")
    base.eval()
    probe.eval()
    assert not torch.isclose(base.training_step(batch(), 0)["loss"], probe.training_step(batch(), 0)["loss"])
    with torch.no_grad():
        old, new = base.eval_step(batch()), probe.eval_step(batch())
    for name in old:
        torch.testing.assert_close(old[name], new[name])
    assert TigerTrainingProbe._compute_loss is Tiger._compute_loss
    assert TigerTrainingProbe.eval_step is Tiger.eval_step


def test_conditional_lookup_normalizes_by_target_mass_and_distinguishes_parents():
    mass = torch.tensor([9.0, 1.0, 1.0, 9.0])
    objective = ConditionalBranchObjective(SIDS, mass, 4, layers=[2], alpha=1, cap=2)
    weights = objective.weights_for(SIDS)
    assert torch.equal(weights[:, 0], torch.ones(4))
    assert weights[1, 1] > weights[0, 1]
    assert weights[2, 1] > weights[3, 1]
    assert weights.max() <= 2
    torch.testing.assert_close((weights[:, 1] * mass).sum() / mass.sum(), torch.tensor(1.0))
    assert not objective.state_dict()


@pytest.mark.parametrize(
    "kwargs", [{"alpha": -1}, {"cap": 0.5}, {"layers": [0]}, {"layers": [2, 2]}, {"alpha": float("nan")}]
)
def test_bad_objective_parameters(kwargs):
    with pytest.raises(ValueError):
        ConditionalBranchObjective(SIDS, torch.ones(4), 4, **kwargs)


def test_unknown_targets_and_zero_frequency():
    objective = ConditionalBranchObjective(SIDS, torch.tensor([1.0, 0.0, 0.0, 0.0]), 4, layers=[2])
    assert torch.equal(objective.weights_for(SIDS), torch.ones(4, 2))
    with pytest.raises(ValueError, match="absent"):
        objective.weights_for(torch.tensor([[2, 0]]))


def test_resume_conflict_rejected_before_model_instantiation():
    with pytest.raises(ValueError, match="resume"):
        TigerTrainingProbe(statistics={}, initialization_checkpoint_path="x", arm="ce", resume_checkpoint_path="y")


def test_strict_initialization_rejects_wrong_checkpoint(tmp_path):
    path = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        TigerTrainingProbe(
            **model_kwargs(),
            initialization_checkpoint_path=str(path),
            arm="ce",
            layers=[2],
            statistics={"semantic_ids": SIDS, "expected_counts": torch.ones(4), "metadata": {}},
        )
