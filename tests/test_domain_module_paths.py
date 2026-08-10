from pathlib import Path

from src.embedding.embedding_aggregator import EmbeddingAggregator
from src.quantization.rqvae.mlp import MLP
from src.quantization.rqvae.normalize_layer import NormalizeLayer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_domain_owned_modules_import_from_domain_paths() -> None:
    assert EmbeddingAggregator.__module__ == "src.embedding.embedding_aggregator"
    assert MLP.__module__ == "src.quantization.rqvae.mlp"
    assert NormalizeLayer.__module__ == "src.quantization.rqvae.normalize_layer"


def test_model_configs_use_domain_module_targets() -> None:
    sem_embeds_config = (PROJECT_ROOT / "configs/model/sem_embeds_inference.yaml").read_text()
    rqvae_config = (PROJECT_ROOT / "configs/model/rqvae_train.yaml").read_text()

    assert "src.embedding.embedding_aggregator.EmbeddingAggregator" in sem_embeds_config
    assert "src.quantization.rqvae.mlp.MLP" in rqvae_config
    assert "src.quantization.rqvae.normalize_layer.NormalizeLayer" in rqvae_config
    assert "src.common.modules" not in sem_embeds_config
    assert "src.common.modules" not in rqvae_config
