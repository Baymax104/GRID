import torch

import src.common.components.eval_metrics as eval_metrics
from src.common.components.eval_metrics import NDCG, Recall
from src.common.metrics import SIDRetrievalMetricGroup


def test_sid_retrieval_metric_group_computes_top_k_metrics():
    group = SIDRetrievalMetricGroup(
        metrics={
            "ndcg": NDCG,
            "recall": Recall,
        },
        top_k_list=[1, 2],
    )

    group.update_from_payload(
        {
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
    )

    computed = group.compute()

    assert set(computed) == {"ndcg@1", "ndcg@2", "recall@1", "recall@2"}
    assert torch.isclose(computed["recall@1"], torch.tensor(0.5))
    assert torch.isclose(computed["recall@2"], torch.tensor(1.0))
    assert torch.isclose(computed["ndcg@1"], torch.tensor(0.5))
    expected_ndcg_at_2 = (torch.tensor(1.0) + torch.tensor(1.0) / torch.log2(torch.tensor(3.0))) / 2.0
    assert torch.isclose(computed["ndcg@2"], expected_ndcg_at_2)


def test_legacy_evaluator_wrappers_are_removed_but_metrics_remain():
    assert not hasattr(eval_metrics, "Evaluator")
    assert not hasattr(eval_metrics, "SIDRetrievalEvaluator")
    assert eval_metrics.NDCG is NDCG
    assert eval_metrics.Recall is Recall
