from functools import partial

import pytest
import torch
from torch.nn import functional as F
from transformers import T5Config, T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Stack

from src.common.configs.model import TrainingModelConfig
from src.data.components.catalog_content import load_catalog_content
from src.data.components.data_models import TigerLabelData, TigerModelInput
from src.data.components.prefix_trace import validate_prefix_trace_bundle
from src.recommendation.tiger.tiger import Tiger
from src.recommendation.tiger_catalog_grounded.catalog import PrefixCatalog
from src.recommendation.tiger_catalog_grounded.module import ARMS, TigerCatalogGrounded

SIDS = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])


def catalog():
    return dict(
        keys=torch.tensor([2, 5, 7, 9]),
        semantic_ids=SIDS.clone(),
        embeddings=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, -1.0]]),
    )


def kwargs():
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
            loss_function=torch.nn.CrossEntropyLoss(), optimizer=partial(torch.optim.Adam, lr=0.001)
        ),
    )


def make(arm="full", **options):
    return TigerCatalogGrounded(**kwargs(), catalog=catalog(), arm=arm, projection_dim=3, **options)


def batch():
    ids = torch.tensor([[0, 1], [1, 0]])
    return TigerModelInput(
        input_ids=ids, attention_mask=torch.ones_like(ids), output_keys=torch.tensor([11, 12])
    ), TigerLabelData(target_ids=SIDS[[1, 2]])


@pytest.mark.parametrize("arm", ARMS)
def test_arms_backward_valid_unique_generation_and_restore(arm):
    torch.manual_seed(17)
    model = make(arm)
    loss = model.training_step(batch(), 0)["loss"]
    assert loss.isfinite()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.isfinite().all(), name
    model.eval()
    inputs, _ = batch()
    with torch.no_grad():
        ids, scores = model.generate(inputs.attention_mask, inputs.input_ids)
    assert ids.shape == (2, 2, 2)
    assert scores.isfinite().all() and (scores > 0).all()
    assert (scores[:, 0] >= scores[:, 1]).all()
    for row in ids:
        assert row.unique(dim=0).shape[0] == 2
    model.catalog.item_indices(ids)
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    restored = make(arm).eval()
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.no_grad():
        restored_ids, restored_scores = restored.generate(inputs.attention_mask, inputs.input_ids)
    torch.testing.assert_close(ids, restored_ids)
    torch.testing.assert_close(scores, restored_scores)


def test_original_exact_baseline_and_extra_initialization_does_not_consume_rng():
    torch.manual_seed(42)
    base = Tiger(**kwargs()).eval()
    state = torch.random.get_rng_state()
    torch.manual_seed(42)
    original = make("original").eval()
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(base.training_step(batch(), 0)["loss"], original.training_step(batch(), 0)["loss"])
    for arm in ("full", "shuffled", "no_aux", "mask_ce", "hybrid"):
        torch.manual_seed(42)
        model = make(arm)
        assert torch.equal(state, torch.random.get_rng_state())
        for name, tensor in base.state_dict().items():
            torch.testing.assert_close(tensor, model.state_dict()[name])


def test_exact_mass_counts_and_jensen_bound():
    bank = PrefixCatalog(catalog(), 4, 2, projection_dim=3, prototypes=1)
    q = torch.nn.functional.normalize(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
    masses = bank.log_mass(q, torch.empty(1, 0, dtype=torch.long), 0.1)[0]
    for token in (0, 1):
        members = bank.sids[:, 0] == token
        mu = bank.features[members].mean(0)
        expected = torch.log(torch.tensor(float(members.sum()))) + (q[0] * mu).sum() / 0.1
        torch.testing.assert_close(masses[token], expected)
        exact = (bank.features[members] @ q[0] / 0.1).logsumexp(0)
        assert masses[token] <= exact + 1e-5
    assert bank.counts_0.sum() == 4 and bank.counts_1.sum() == 4
    leaves = bank.log_mass(q, torch.tensor([[0]]), 0.1)[0, :2]
    torch.testing.assert_close(leaves, bank.features[:2] @ q[0] / 0.1)
    assert torch.isneginf(masses[2:]).all()


def test_content_projection_initialization_does_not_reseed_cuda(monkeypatch):
    options = kwargs()
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    TigerCatalogGrounded(**options, catalog=catalog(), projection_dim=3)
    assert calls == []


def test_multiple_prototypes_preserve_mass_and_satisfy_radius_error_bound():
    values = dict(
        keys=torch.arange(16),
        semantic_ids=torch.tensor([[i // 8, i % 8] for i in range(16)]),
        embeddings=torch.randn(16, 5, generator=torch.Generator().manual_seed(3)),
    )
    bank = PrefixCatalog(values, 8, 2, projection_dim=4, prototypes=2)
    single = PrefixCatalog(values, 8, 2, projection_dim=4, prototypes=1)
    queries = F.normalize(torch.randn(8, 4, generator=torch.Generator().manual_seed(4)), dim=-1)
    prefixes = torch.empty(8, 0, dtype=torch.long)
    masses = bank.log_mass(queries, prefixes, .1)
    lower = single.log_mass(queries, prefixes, .1)
    for token in (0, 1):
        members = bank.sids[:, 0] == token
        slots = bank.slots_0[token]
        slots = slots[slots >= 0]
        count = bank.counts_0[slots]
        assert count.sum() == 8 and (count > 0).all()
        weighted_mean = (bank.means_0[slots] * count[:, None]).sum(0) / count.sum()
        torch.testing.assert_close(weighted_mean, bank.features[members].mean(0))
        exact = (queries @ bank.features[members].T / .1).logsumexp(-1)
        assert (masses[:, token] >= lower[:, token] - 1e-5).all()
        assert (exact >= masses[:, token] - 1e-5).all()
        assert (exact - masses[:, token] <= bank.radii_0[slots].max() / .1 + 1e-5).all()


def test_mixture_probability_and_teacher_beam_score_agree():
    model = make().eval()
    query = torch.nn.functional.normalize(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
    prefixes = torch.empty(1, 0, dtype=torch.long)
    logits = torch.tensor([[0.2, 0.7, 100.0, 10.0]], requires_grad=True)
    actual = model.conditional_log_probs(logits, prefixes, query)
    expected = 0.9 * logits[:, :2].softmax(-1) + 0.1 * model.catalog.log_mass(query, prefixes, 0.1)[:, :2].softmax(-1)
    torch.testing.assert_close(actual[:, :2].exp(), expected)
    assert torch.isneginf(actual[:, 2:]).all()
    inputs, _ = batch()
    with torch.no_grad():
        ids, scores = model.generate(inputs.attention_mask, inputs.input_ids)
        for candidate in range(2):
            teacher = model(inputs.attention_mask, inputs.input_ids, ids[:, candidate])
            global_ids = ids[:, candidate] + torch.tensor([0, 4])
            teacher_score = teacher.gather(-1, global_ids.unsqueeze(-1)).squeeze(-1).sum(-1).exp()
            torch.testing.assert_close(scores[:, candidate], teacher_score, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("arm", ["original", "mask_ce", "full"])
def test_trace_passes_existing_schema_and_predict_identity(arm):
    metadata = dict(
        data_split="evaluation",
        beam_width=2,
        num_hierarchies=2,
        codebook_size=4,
        trace_mode="teacher_forcing_and_constrained_beam",
        checkpoint_reference="fixture",
        semantic_id_reference="fixture",
    )
    model = make(arm, trace_prefix_survival=True, prefix_trace_metadata=metadata).eval()
    with torch.no_grad():
        output = model.predict_step(batch())
    payload = {"keys": output.keys, **output.auxiliary["prefix_trace"]}
    validate_prefix_trace_bundle(payload)
    assert payload["trace"]["target_prefix_survived"].shape == (2, 2)


def test_narrow_root_beam_uses_inactive_slots_without_invalid_or_duplicate_items():
    values = catalog()
    values["semantic_ids"] = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]])
    options = kwargs()
    options.update(semantic_ids=values["semantic_ids"], top_k_for_generation=4)
    model = TigerCatalogGrounded(**options, catalog=values, projection_dim=3).eval()
    inputs, _ = batch()
    with torch.no_grad():
        ids, scores = model.generate(inputs.attention_mask, inputs.input_ids)
    assert ids.shape == (2, 4, 2) and scores.isfinite().all()
    assert all(row.unique(dim=0).shape[0] == 4 for row in ids)
    model.catalog.item_indices(ids)


def test_checkpoint_rejects_missing_changed_arm_and_changed_content():
    model = make()
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="contract mismatch"):
        model.on_load_checkpoint({})
    with pytest.raises(ValueError, match="contract mismatch"):
        make("no_aux").on_load_checkpoint(checkpoint)
    changed = catalog()
    changed["embeddings"][0, 0] += 0.1
    other = TigerCatalogGrounded(**kwargs(), catalog=changed, projection_dim=3)
    with pytest.raises(ValueError, match="contract mismatch"):
        other.on_load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="beam-only trace"):
        make("hybrid", trace_prefix_survival=True)


def test_keyed_loader_aligns_permutations_and_rejects_missing_keys(tmp_path):
    sid_path, embed_path = tmp_path / "sids.pt", tmp_path / "embeddings.pt"
    values = catalog()
    order = torch.tensor([2, 0, 3, 1])
    torch.save(dict(keys=values["keys"], predictions=values["semantic_ids"]), sid_path)
    torch.save(dict(keys=values["keys"][order], predictions=values["embeddings"][order]), embed_path)
    loaded = load_catalog_content(str(sid_path), str(embed_path))
    torch.testing.assert_close(loaded["embeddings"], values["embeddings"])
    torch.save(dict(keys=values["keys"][:3], predictions=values["embeddings"][:3]), embed_path)
    with pytest.raises(ValueError, match="missing"):
        load_catalog_content(str(sid_path), str(embed_path))


@pytest.mark.parametrize("failure", ["duplicate_sid", "fractional", "nonfinite", "overflow"])
def test_bank_rejects_invalid_inputs(failure):
    values = catalog()
    if failure == "duplicate_sid":
        values["semantic_ids"][0] = values["semantic_ids"][1]
    elif failure == "fractional":
        values["semantic_ids"] = values["semantic_ids"].float() + 0.1
    elif failure == "nonfinite":
        values["embeddings"][0, 0] = torch.nan
    with pytest.raises(ValueError):
        PrefixCatalog(values, 2**40 if failure == "overflow" else 4, 2)
