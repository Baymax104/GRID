import torch

from src.recommendation.tiger.metric_adapters import sid_retrieval_inputs


def test_sid_retrieval_inputs_converts_tiger_payload_to_retrieval_kwargs():
    payload = {
        "marginal_probs": torch.tensor(
            [
                [0.9, 0.1],
                [0.8, 0.2],
            ]
        ),
        "generated_ids": torch.tensor(
            [
                [[1, 1], [2, 2]],
                [[3, 3], [4, 4]],
            ]
        ),
        "labels": torch.tensor(
            [
                [1, 1],
                [4, 4],
            ]
        ),
    }

    inputs = sid_retrieval_inputs(payload)

    assert set(inputs) == {"preds", "target", "indexes"}
    assert torch.equal(inputs["preds"], torch.tensor([0.9, 0.1, 0.8, 0.2]))
    assert torch.equal(inputs["target"], torch.tensor([True, False, False, True]))
    assert torch.equal(inputs["indexes"], torch.tensor([0, 0, 1, 1]))
    assert inputs["preds"].device == payload["marginal_probs"].device
    assert inputs["target"].device == payload["marginal_probs"].device
    assert inputs["indexes"].device == payload["marginal_probs"].device
