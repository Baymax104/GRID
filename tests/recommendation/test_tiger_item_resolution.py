from functools import partial

import pytest
import torch
from transformers import T5Config, T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Stack

from src.common.configs.model import TrainingModelConfig
from src.data.components.data_models import TigerLabelData, TigerModelInput
from src.data.components.item_resolution import validate_resolution_calibration, validate_resolution_trace
from src.recommendation.tiger_item_resolution import TigerItemResolution
from src.recommendation.tiger_item_resolution.module import ARMS, RESOLUTION_ARMS


def catalog():
    return dict(
        keys=torch.arange(16) * 3,
        semantic_ids=torch.tensor([[i // 8, i // 4 % 2, i // 2 % 2, i % 2] for i in range(16)]),
        embeddings=torch.randn(16, 6, generator=torch.Generator().manual_seed(3)),
    )


def make(arm="mir", **options):
    cfg = dict(vocab_size=4, d_model=8, num_heads=2, d_ff=16, d_kv=4, num_layers=1, dropout_rate=0.0)
    values = catalog()
    defaults = dict(
        encoder=T5EncoderModel(T5Config(**cfg)),
        decoder=T5Stack(T5Config(**cfg, is_decoder=True, is_encoder_decoder=False)),
        semantic_ids=values["semantic_ids"],
        num_hierarchies=4,
        codebook_size=4,
        embedding_dim=8,
        top_k_for_generation=4,
        should_check_prefix=True,
        catalog=values,
        arm=arm,
        projection_dim=4,
        max_bucket=8,
        warmup_steps=0,
        max_states=128,
        max_item_scores=512,
        expansion_batch_size=2,
        data_split="evaluation",
        training_model_config=TrainingModelConfig(
            loss_function=torch.nn.CrossEntropyLoss(), optimizer=partial(torch.optim.Adam, lr=0.001)
        ),
    )
    defaults.update(options)
    return TigerItemResolution(**defaults)


def batch():
    sids = catalog()["semantic_ids"]
    return TigerModelInput(
        input_ids=sids[[0, 9]].clone(),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        output_keys=torch.tensor([31, 29]),
    ), TigerLabelData(target_ids=sids[[3, 15]])


@pytest.mark.parametrize("arm", ARMS)
def test_training_generation_checkpoint_and_trace(arm):
    torch.manual_seed(42)
    model = make(arm, trace_resolution=True)
    loss = model.training_step(batch(), 0)["loss"]
    assert loss.isfinite()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(gradient.isfinite().all() for gradient in gradients)
    model.eval()
    with torch.no_grad():
        output = model.predict_step(batch())
    assert output.predictions.shape == (2, 4, 4)
    model.catalog.item_indices(output.predictions)
    payload = dict(output.auxiliary["item_resolution_trace"], keys=output.keys)
    validate_resolution_trace(payload)
    checkpoint = dict(state_dict=model.state_dict())
    model.on_save_checkpoint(checkpoint)
    restored = make(arm, max_states=129, trace_resolution=True).eval()
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad():
        actual = restored.predict_step(batch())
    torch.testing.assert_close(output.predictions, actual.predictions)


@pytest.mark.parametrize("arm", sorted(RESOLUTION_ARMS))
def test_complete_search_equals_exact_marginal_and_normalizes(arm):
    model = make(arm, top_k_for_generation=16).eval()
    inputs, _ = batch()
    with torch.no_grad():
        encoded, mask = model.encoder(input_ids=inputs.input_ids[:1], attention_mask=inputs.attention_mask[:1])
        log_p = model.objective(encoded.expand(16, -1, -1), mask.expand(16, -1), model.catalog.sids)[
            "target_log_probability"
        ]
        torch.testing.assert_close(log_p.exp().sum(), torch.tensor(1.0), atol=2e-6, rtol=2e-6)
        ids, scores, trace = model.generate_encoded(encoded, mask)
        order = model.catalog.item_indices(ids)[0]
        torch.testing.assert_close(scores[0], log_p.exp()[order], atol=2e-6, rtol=2e-6)
        assert trace["remaining_mass"].max() < 1e-6
        assert trace["topk_certified"].all()


def test_budget_bounds_are_valid_and_do_not_change_model_probability():
    model = make(max_states=4, expansion_batch_size=1).eval()
    inputs, _ = batch()
    with torch.no_grad():
        encoded, mask = model.encoder(input_ids=inputs.input_ids[:1], attention_mask=inputs.attention_mask[:1])
        log_p = model.objective(encoded.expand(16, -1, -1), mask.expand(16, -1), model.catalog.sids)[
            "target_log_probability"
        ]
        ids, scores, trace = model.generate_encoded(encoded, mask)
        actual = log_p.exp()[model.catalog.item_indices(ids)[0]]
        assert (actual + 1e-6 >= scores[0]).all()
        assert (actual <= scores[0] + trace["remaining_mass"][0] + 1e-6).all()
        torch.testing.assert_close(trace["remaining_mass"] + trace["resolved_total_mass"], torch.ones(1))
        assert trace["states_evaluated"].max() <= 4
    model.max_states = 1
    with pytest.raises(RuntimeError, match="fewer than K"):
        model.generate_encoded(encoded, mask)


def test_prefix_states_and_gate_do_not_read_target_suffix():
    model = make().eval()
    inputs, _ = batch()
    encoded, mask = model.encoder(
        input_ids=inputs.input_ids[:1].expand(2, -1), attention_mask=inputs.attention_mask[:1].expand(2, -1)
    )
    prefixes = torch.tensor([[0, 0, 0], [0, 1, 1]])
    hidden = model.states(encoded, mask, prefixes)
    torch.testing.assert_close(hidden[0, :2], hidden[1, :2])
    nodes = model.catalog.nodes(prefixes[:, :1])
    gates = model.gates(hidden[:, 1], nodes, 1)
    torch.testing.assert_close(gates[0], gates[1])
    lp, _, _ = model.resolve_log(hidden[:, 1], nodes, 1, model.item_vectors())
    torch.testing.assert_close(lp[0], lp[1])


def test_checkpoint_rejects_arm_and_content_change():
    model = make()
    checkpoint = dict(state_dict=model.state_dict())
    model.on_save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="contract mismatch"):
        make("depth_gate").on_load_checkpoint(checkpoint)
    changed = catalog()
    changed["embeddings"][0, 0] += 0.1
    with pytest.raises(ValueError, match="contract mismatch"):
        make(catalog=changed).on_load_checkpoint(checkpoint)


def test_warmup_detaches_reporting_tensors_for_ddp():
    model = make(warmup_steps=5)
    output = model.training_step(batch(), 0)
    assert all(not value.requires_grad for name, value in output.items() if name != "loss")
    output["loss"].backward()
    assert all(parameter.grad is None for parameter in model.gate_head.parameters())


def test_calibration_and_wide_are_training_only_and_checkpoint_matched():
    model = make("hybrid", inference_policy="calibrate", data_split="training").eval()
    checkpoint = dict(state_dict=model.state_dict())
    model.on_save_checkpoint(checkpoint)
    model.on_load_checkpoint(checkpoint)
    with torch.no_grad():
        output = model.predict_step(batch())
    payload = dict(output.auxiliary["item_resolution_calibration"], keys=output.keys)
    validate_resolution_calibration(payload)
    calibration = {**payload["metadata"], "thresholds": payload["trace"]["entropy"].mean(0)}
    wide = make("hybrid", inference_policy="wide", calibration=calibration, trace_resolution=True).eval()
    wide.on_load_checkpoint(checkpoint)
    wide.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad():
        result = wide.predict_step(batch())
    validate_resolution_trace(dict(result.auxiliary["item_resolution_trace"], keys=result.keys))
    assert not result.auxiliary["item_resolution_trace"]["trace"]["probability_bound_valid"].any()
    payload["metadata"]["source_split"] = "testing"
    with pytest.raises(ValueError, match="training split"):
        validate_resolution_calibration(payload)
    calibration["checkpoint_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        wide.on_load_checkpoint(checkpoint)


def test_resolution_trace_round_trip_in_shared_writer(tmp_path, monkeypatch):
    from src.common.writers.auxiliary_tensor_writer import AuxiliaryTensorWriter

    monkeypatch.setattr("src.common.writers.auxiliary_tensor_writer.sync_file", lambda _: None)
    model = make(trace_resolution=True).eval()
    with torch.no_grad():
        output = model.predict_step(batch())
    writer = AuxiliaryTensorWriter(
        str(tmp_path), "item_resolution_trace", "item_resolution_trace.pt", validator=validate_resolution_trace
    )
    writer.global_rank = 0
    writer.buffer = [output]
    writer.flush_buffer()
    path, _ = writer._merge_files()
    bundle = torch.load(path, weights_only=False)
    validate_resolution_trace(bundle)
    assert bundle["keys"].tolist() == [29, 31]
    torch.testing.assert_close(
        bundle["trace"]["topk_scores"], output.auxiliary["item_resolution_trace"]["trace"]["topk_scores"][[1, 0]]
    )


@pytest.mark.parametrize("bias", [-80.0, 80.0])
def test_saturated_gates_keep_finite_loss_and_gradients(bias):
    model = make()
    with torch.no_grad():
        model.gate_head[-1].bias.fill_(bias)
    loss = model.training_step(batch(), 0)["loss"]
    loss.backward()
    assert loss.isfinite()
    assert all(parameter.grad.isfinite().all() for parameter in model.parameters() if parameter.grad is not None)


def test_forced_route_for_large_buckets_and_terminal_capacity_validation():
    model = make(max_bucket=4).eval()
    inputs, labels = batch()
    encoded, mask = model.encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    hidden = model.states(encoded, mask, labels.target_ids[:, :1])[:, -1]
    nodes = model.catalog.nodes(labels.target_ids[:, :1])
    assert (model.gates(hidden, nodes, 1) == 0).all()
    output = model.training_step(batch(), 0)
    output["loss"].backward()
    assert output["loss"].isfinite()
    with pytest.raises(ValueError, match="Terminal semantic bucket"):
        make(max_bucket=1)


def test_flat_first_layer_can_complete_candidates_without_hidden_extra_budget():
    values = dict(
        keys=torch.arange(256),
        semantic_ids=torch.tensor([[i // 2, 0, 0, i % 2] for i in range(256)]),
        embeddings=torch.randn(256, 6, generator=torch.Generator().manual_seed(3)),
    )
    model = make(
        "depth2",
        catalog=values,
        semantic_ids=values["semantic_ids"],
        codebook_size=128,
        max_bucket=2,
        max_states=16,
        expansion_batch_size=8,
    ).eval()
    with torch.no_grad():
        model.decoder.lm_head.weight.zero_()
        inputs = values["semantic_ids"][:1]
        encoded, mask = model.encoder(input_ids=inputs, attention_mask=torch.ones_like(inputs))
        ids, scores, trace = model.generate_encoded(encoded, mask)
    assert ids.shape == (1, 4, 4) and (scores > 0).all()
    assert int(trace["states_evaluated"][0]) <= 16
    torch.testing.assert_close(trace["remaining_mass"] + trace["resolved_total_mass"], torch.ones(1))


@pytest.mark.parametrize("arm", ARMS)
def test_lightweight_validation_matches_full_search(arm, monkeypatch):
    from src.recommendation.tiger_item_resolution import search

    model = make(arm).eval()
    inputs, labels = batch()
    with torch.no_grad():
        encoded, mask = model.encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        expected_ids, expected_scores, _ = model.generate_encoded(encoded, mask)

        def unused_certificate(*args, **kwargs):
            pytest.fail("Lightweight evaluation must not compute a certificate.")

        monkeypatch.setattr(search, "_frontier_upper", unused_certificate)
        if arm in RESOLUTION_ARMS:
            monkeypatch.setattr(search, "_empty_trace", unused_certificate)
        actual_ids, actual_scores, trace = model.generate_encoded(encoded, mask, collect_trace=False)
        assert trace == {}
        assert torch.equal(actual_ids, expected_ids)
        assert torch.equal(actual_scores, expected_scores)
        output = model.eval_step((inputs, labels))
        assert torch.equal(output["generated_ids"], expected_ids)
        assert torch.equal(output["marginal_probs"], expected_scores)
        prediction = model.predict_step((inputs, labels))
        assert prediction.auxiliary == {}
        assert torch.equal(prediction.predictions, expected_ids)


def test_batched_frontier_upper_matches_member_reference_without_member_queries(monkeypatch):
    from src.recommendation.tiger_item_resolution.search import _frontier_upper

    model = make()
    directory = model.catalog
    node = lambda prefix: int(directory.nodes(torch.tensor([prefix]))[0])  # noqa: E731
    frontier = [{0: 1.0}, {node([0]): 0.125, node([1, 0]): 0.25, node([1, 1, 0]): 0.0625}, {}]
    lower = torch.rand(3, len(directory.keys))
    reference = lower.clone()
    for user, queue in enumerate(frontier):
        for prefix_node, mass in queue.items():
            members, valid = directory.members(torch.tensor([prefix_node]))
            reference[user, members[0, valid[0]]] += mass

    def forbidden_members(*args):
        pytest.fail("A batch certificate must not query members node by node.")

    monkeypatch.setattr(directory, "members", forbidden_members)
    actual = _frontier_upper(directory, lower, frontier)
    assert torch.equal(actual, reference)


def test_derived_catalog_index_preserves_checkpoint_and_device_contract():
    model = make()
    state = model.state_dict()
    assert "catalog.item_ancestors" not in state
    assert "catalog.item_ancestors" in dict(model.named_buffers())
    rebuilt = make()
    rebuilt.load_state_dict(state, strict=True)
    assert torch.equal(model.catalog.item_ancestors, rebuilt.catalog.item_ancestors)
    assert model.contract == rebuilt.contract
    assert model.catalog.node_counts == tuple(model.catalog.counts.tolist())


def test_wide_frontier_does_not_trigger_per_node_member_queries(monkeypatch):
    values = dict(
        keys=torch.arange(256),
        semantic_ids=torch.tensor([[i // 2, 0, 0, i % 2] for i in range(256)]),
        embeddings=torch.randn(256, 6, generator=torch.Generator().manual_seed(3)),
    )
    model = make(
        "mir",
        catalog=values,
        semantic_ids=values["semantic_ids"],
        codebook_size=128,
        max_bucket=2,
        max_states=16,
        expansion_batch_size=8,
    ).eval()
    original_members = model.catalog.members
    calls = []

    def counted(nodes):
        calls.append(len(nodes))
        return original_members(nodes)

    monkeypatch.setattr(model.catalog, "members", counted)
    with torch.no_grad():
        model.decoder.lm_head.weight.zero_()
        inputs = values["semantic_ids"][:2]
        encoded, mask = model.encoder(input_ids=inputs, attention_mask=torch.ones_like(inputs))
        _, _, trace = model.generate_encoded(encoded, mask)
    # 128 个第一层分支仍留有大量前沿，但成员查询只发生在批量局部解析。
    assert len(calls) <= 8
    assert (trace["remaining_mass"] > 0).all()
    torch.testing.assert_close(trace["remaining_mass"] + trace["resolved_total_mass"], torch.ones(2))


@pytest.mark.parametrize("arm", ["mir", "depth2"])
def test_exact_audit_distribution_matches_exhaustive_search_and_chunks(arm):
    from src.recommendation.tiger_item_resolution.audit import exact_catalog_distribution

    model = make(arm).eval()
    inputs, _ = batch()
    with torch.no_grad():
        encoded, mask = model.encoder(input_ids=inputs.input_ids[:1], attention_mask=inputs.attention_mask[:1])
        logp, resp = exact_catalog_distribution(model, encoded, mask, 3)
        whole, other_resp = exact_catalog_distribution(model, encoded, mask, 16)
        sids, scores, trace = model.generate_encoded(encoded, mask)
    torch.testing.assert_close(logp, whole, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(resp, other_resp, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(logp.exp().sum(), torch.tensor(1.0))
    torch.testing.assert_close(scores[0], logp[model.catalog.item_indices(sids)[0]].exp())
    assert trace["remaining_mass"].item() == 0


def test_audit_sampling_is_key_only_order_independent_and_rejects_duplicates():
    from src.data.components.item_resolution_audit import select_audit_rows

    rows = [{"user_id": torch.tensor([i]), "target_ids": torch.tensor([i % 2])} for i in range(50)]
    expected = select_audit_rows(rows, 7, 42)
    changed = [{**r, "target_ids": torch.tensor([999])} for r in reversed(rows)]
    actual = select_audit_rows(changed, 7, 42)
    assert [int(r["user_id"]) for r in expected] == [int(r["user_id"]) for r in actual]
    with pytest.raises(ValueError, match="Duplicate"):
        select_audit_rows(rows + rows[:1], 7, 42)
    with pytest.raises(ValueError, match="fewer"):
        select_audit_rows(rows, 51, 42)


def test_audit_payload_round_trip_and_no_target_injection(monkeypatch, tmp_path):
    from src.common.writers.auxiliary_tensor_writer import AuxiliaryTensorWriter
    from src.data.components.item_resolution_audit import validate_score_search_audit
    from src.recommendation.tiger_item_resolution.audit import TigerItemResolutionAudit

    monkeypatch.setitem(make.__globals__, "TigerItemResolution", TigerItemResolutionAudit)
    model = make(audit_users=1, audit_chunk_size=4, audit_state_budgets=(16, 64)).eval()
    checkpoint = dict(state_dict=model.state_dict())
    model.on_save_checkpoint(checkpoint)
    model.on_load_checkpoint(checkpoint)
    inp, labels = batch()
    one = TigerModelInput(
        input_ids=inp.input_ids[:1], attention_mask=inp.attention_mask[:1], output_keys=inp.output_keys[:1]
    )
    output = model.predict_step((one, TigerLabelData(target_ids=labels.target_ids[:1])))
    changed = model.predict_step((one, TigerLabelData(target_ids=labels.target_ids[1:2])))
    a, b = output.auxiliary["item_resolution_audit"], changed.auxiliary["item_resolution_audit"]
    validate_score_search_audit(dict(a, keys=output.keys))
    assert model.max_states == 128
    for field in [
        "exact_log_probability",
        "search_topk_keys",
        "search_topk_scores",
        "input_sha256",
        "global_depth_mass",
    ]:
        assert torch.equal(a["trace"][field], b["trace"][field])
    assert torch.equal(output.predictions, changed.predictions)
    monkeypatch.setattr("src.common.writers.auxiliary_tensor_writer.sync_file", lambda _: None)
    writer = AuxiliaryTensorWriter(
        str(tmp_path), "item_resolution_audit", "item_resolution_audit.pt", validator=validate_score_search_audit
    )
    writer.global_rank, writer.buffer = 0, [output]
    writer.flush_buffer()
    path, _ = writer._merge_files()
    validate_score_search_audit(torch.load(path, weights_only=False))
    with pytest.raises(RuntimeError, match="frozen"):
        model.training_step(batch(), 0)
