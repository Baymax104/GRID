import copy
import random

import pytest
import torch
import torch.nn.functional as F

import src.quantization.letter.tokenizer as tokenizer_module
from src.quantization.letter.tokenizer import LetterTokenizer, diversity_loss, sinkhorn_assign


def legacy_diversity_loss(codes, ids, labels, positives=None):
    """保留修复前采样作为差分 oracle。"""
    if positives is None:
        groups = {int(group): (labels == group).nonzero().flatten().tolist() for group in labels.unique()}
        sampled = []
        for index in ids.tolist():
            choices = [other for other in groups[int(labels[index])] if other != index]
            if not choices:
                raise ValueError("Diversity positive cluster must contain a non-self code.")
            sampled.append(random.choice(choices))
        positives = torch.tensor(sampled, device=codes.device)
    if (positives == ids).any() or not torch.equal(labels[positives], labels[ids]):
        raise ValueError("Diversity positives must be distinct codes in the same cluster.")
    logits = (codes[ids] @ codes.T).scatter(1, ids[:, None], -1e12)
    return F.cross_entropy(logits, positives)


@pytest.fixture
def preserve_sampling_rng():
    state = random.getstate()
    yield
    random.setstate(state)


@pytest.mark.parametrize("seed", [42, 200, 2026])
def test_diversity_matches_legacy_sampling_rng_and_gradient(seed, preserve_sampling_rng):
    codes = torch.randn(8, 4, generator=torch.Generator().manual_seed(seed), requires_grad=True)
    reference = codes.detach().clone().requires_grad_()
    labels = torch.tensor([8, 1, 8, 7, 1, 8, 7, 1])
    ids = torch.tensor([0, 7, 3, 7, 4, 5, 0, 6])
    random.seed(seed)
    expected = legacy_diversity_loss(reference, ids, labels)
    expected_rng = random.getstate()
    expected.backward()
    random.seed(seed)
    actual = diversity_loss(codes, ids, labels)
    actual.backward()
    assert random.getstate() == expected_rng
    assert torch.equal(actual, expected)
    assert torch.equal(codes.grad, reference.grad)


def test_diversity_reads_sampling_inputs_in_batches(monkeypatch, preserve_sampling_rng):
    codes = torch.randn(8, 4)
    ids = torch.arange(8).repeat(128)
    labels = torch.tensor([8, 1, 8, 7, 1, 8, 7, 1])
    original = torch.Tensor.tolist
    reads = []

    def counted(tensor):
        reads.append(tensor)
        return original(tensor)

    def reject_scalar_read(tensor):
        raise AssertionError("Diversity sampling must not read individual tensor scalars.")

    monkeypatch.setattr(torch.Tensor, "tolist", counted)
    monkeypatch.setattr(torch.Tensor, "__int__", reject_scalar_read)
    assert diversity_loss(codes, ids, labels).isfinite()
    assert len(reads) == 2
    assert any(tensor is ids for tensor in reads)
    assert any(tensor is labels for tensor in reads)


def test_explicit_diversity_positives_skip_sampling(monkeypatch, preserve_sampling_rng):
    def reject_sampling(*args):
        raise AssertionError("Explicit positives must not sample or read CPU lists.")

    monkeypatch.setattr(random, "choice", reject_sampling)
    monkeypatch.setattr(torch.Tensor, "tolist", reject_sampling)
    state = random.getstate()
    codes = torch.eye(4, requires_grad=True)
    ids, labels = torch.tensor([0, 2]), torch.tensor([0, 0, 1, 1])
    assert diversity_loss(codes, ids, labels, torch.tensor([1, 3])).isfinite()
    assert random.getstate() == state
    with pytest.raises(ValueError, match="distinct"):
        diversity_loss(codes, ids, labels, torch.tensor([2, 1]))


def test_tokenizer_optimizer_trajectory_matches_legacy(monkeypatch, preserve_sampling_rng):
    model = small_model()
    reference = copy.deepcopy(model)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=0.001) for m in (reference, model)]
    features, cf = torch.tensor([[0.3, 0.7], [1.2, 1.1]]), torch.tensor([[0.7, 0.2], [0.1, 0.8]])
    for step in range(3):
        results, rngs = [], []
        for current, optimizer, sampling in zip(
            (reference, model), optimizers, (legacy_diversity_loss, diversity_loss), strict=True
        ):
            monkeypatch.setattr(tokenizer_module, "diversity_loss", sampling)
            random.seed(42 + step)
            optimizer.zero_grad(set_to_none=True)
            result = current(features, cf)
            result["loss"].backward()
            rngs.append(random.getstate())
            optimizer.step()
            results.append(result)
        assert rngs[0] == rngs[1]
        assert all(torch.equal(results[0][name], results[1][name]) for name in results[0])
        for a, b in zip(reference.parameters(), model.parameters(), strict=True):
            assert torch.equal(a, b) and torch.equal(a.grad, b.grad)
        states = [optimizer.state_dict()["state"] for optimizer in optimizers]
        for parameter in states[0]:
            assert all(torch.equal(states[0][parameter][name], states[1][parameter][name]) for name in states[0][parameter])


def small_model():
    model = LetterTokenizer(
        2,
        latent_dim=2,
        codebook_size=4,
        num_layers=2,
        hidden_sizes=(),
        num_groups=2,
        sk_epsilons=(0, 0),
        alpha=0.1,
        beta=0.01,
    )
    model.initialized.fill_(True)
    model.group_labels.copy_(torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]))
    with torch.no_grad():
        model.codebooks.copy_(
            torch.tensor(
                [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6]]]
            )
        )
    return model


def test_official_loss_and_ste_gradient_reference():
    torch.manual_seed(8)
    model = small_model()
    features = torch.tensor([[0.3, 0.7], [1.2, 1.1]])
    cf = torch.tensor([[0.7, 0.2], [0.1, 0.8]])
    latent = model.encoder(features)
    ids = model.quantize(latent, use_sk=False, compute_loss=False)[1]
    positives = [ids[:, level] ^ 1 for level in range(2)]
    result = model(features, cf, positives)
    reference = copy.deepcopy(model)
    residual = reference.encoder(features)
    total = torch.zeros_like(residual)
    losses = []
    for level in range(2):
        codes = reference.codebooks[level]
        distances = ((residual[:, None] - codes[None]) ** 2).sum(-1)
        assigned = distances.argmin(-1)
        values = codes[assigned]
        logits = values @ codes.T
        logits = logits.scatter(1, assigned[:, None], -1e12)
        div = F.cross_entropy(logits, positives[level])
        losses.append(F.mse_loss(values, residual.detach()) + 0.25 * F.mse_loss(values.detach(), residual) + 0.01 * div)
        values = residual + (values - residual).detach()
        total = total + values
        residual = residual - values
    loss = F.mse_loss(reference.decoder(total), features) + torch.stack(losses).mean()
    loss = loss + 0.1 * F.cross_entropy(total @ cf.T, torch.arange(2))
    torch.testing.assert_close(result["loss"], loss)
    result["loss"].backward()
    loss.backward()
    for first, second in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad)


def test_cf_gradient_boundary_and_pure_encoding():
    model = small_model()
    features, cf = torch.randn(3, 2), torch.randn(3, 2)
    result = model(features, cf)
    gradient = torch.autograd.grad(result["cf_loss"], model.codebooks, allow_unused=True)[0]
    assert gradient is None
    encoded = model.encode(features)
    model.group_labels.fill_(-1)
    torch.testing.assert_close(model.encode(features), encoded)
    restored = small_model()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.encode(features), encoded)


def test_diversity_rejects_self_and_singleton():
    codes = torch.eye(4)
    with pytest.raises(ValueError, match="distinct"):
        diversity_loss(codes, torch.tensor([0]), torch.tensor([0, 0, 1, 1]), torch.tensor([0]))
    with pytest.raises(ValueError, match="non-self"):
        diversity_loss(codes, torch.tensor([0]), torch.arange(4))


def test_sinkhorn_and_bounded_collision():
    assert sinkhorn_assign(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), 0.1, 50).tolist() == [0, 1]
    model = small_model()
    with pytest.raises(ValueError, match="unresolved collisions"):
        model.unique_codes(torch.zeros(5, 2), max_rounds=2)


def test_uninitialized_fails():
    model = small_model()
    model.initialized.fill_(False)
    with pytest.raises(ValueError, match="initialized"):
        model.encode(torch.ones(2, 2))


def test_residual_collision_assignment_includes_occupied_neighbors():
    from itertools import permutations

    model = small_model()
    with torch.no_grad():
        model.encoder[1].weight.copy_(torch.eye(2))
        model.encoder[1].bias.zero_()
    features = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.21, 0.21]])
    nearest = model.encode(features)
    assert nearest[:, 0].tolist() == [0, 0, 0]
    result = model.unique_codes(features, max_rounds=0)
    assert result.unique(dim=0).shape[0] == 3
    torch.testing.assert_close(result[:, :-1], nearest[:, :-1])
    costs = ((features[:, None] - model.codebooks[-1][None]) ** 2).sum(-1)
    actual = costs[torch.arange(3), result[:, -1]].sum().item()
    optimum = min(sum(costs[i, j].item() for i, j in enumerate(p)) for p in permutations(range(4), 3))
    assert actual == pytest.approx(optimum)
    torch.testing.assert_close(model.unique_codes(features, max_rounds=0), result)
