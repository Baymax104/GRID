import src.common.components.eval_metrics as eval_metrics
from src.common.components.eval_metrics import NDCG, Recall


def test_legacy_evaluator_wrappers_are_removed_but_metrics_remain():
    assert not hasattr(eval_metrics, "Evaluator")
    assert not hasattr(eval_metrics, "SIDRetrievalEvaluator")
    assert eval_metrics.NDCG is NDCG
    assert eval_metrics.Recall is Recall
