import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import OmegaConf

import src.utils.distributed as distributed_utils
from src.common.configs.model import TrainingModelConfig
from src.common.loss.weighted_squared_error import WeightedSquaredError
from src.quantization.rkmeans.kmeans_layer import KMeansLayer, _kmeans_plus_plus_init
from src.quantization.rkmeans.residual_kmeans import ResidualKMeans
from src.quantization.rqvae.residual_quantization_vae import ResidualQuantizationVAE
from src.quantization.rvq.residual_vector_quantization import ResidualVectorQuantization
from src.quantization.rvq.vector_quantization_layer import VectorQuantizationLayer

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def create_kmeans_layer(**kwargs) -> KMeansLayer:
    return KMeansLayer(**kwargs)


def create_training_model_config(**kwargs) -> TrainingModelConfig:
    kwargs.setdefault("loss_function", WeightedSquaredError())
    return TrainingModelConfig(**kwargs)


def create_residual_kmeans(**kwargs) -> ResidualKMeans:
    n_clusters = kwargs.setdefault("n_clusters", 2)
    n_features = kwargs.setdefault("n_features", 2)
    init_buffer_size = kwargs.pop("init_buffer_size", 4)
    return ResidualKMeans(
        sub_layer=lambda: KMeansLayer(
            n_clusters=n_clusters,
            n_features=n_features,
            init_buffer_size=init_buffer_size,
        ),
        training_model_config=create_training_model_config(),
        **kwargs,
    )


def create_residual_vector_quantization(**kwargs) -> ResidualVectorQuantization:
    n_clusters = kwargs.setdefault("n_clusters", 2)
    n_features = kwargs.setdefault("n_features", 2)
    init_buffer_size = kwargs.pop("init_buffer_size", 4)
    return ResidualVectorQuantization(
        sub_layer=lambda: VectorQuantizationLayer(
            n_clusters=n_clusters,
            n_features=n_features,
            init_buffer_size=init_buffer_size,
        ),
        training_model_config=create_training_model_config(),
        **kwargs,
    )


def test_kmeans_layer_buffers_until_init_buffer_is_full():
    layer = create_kmeans_layer(n_clusters=2, n_features=2, init_buffer_size=4)
    residuals = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    ids, embeddings, batch_cluster_counts, batch_cluster_sums = layer(residuals)

    assert ids.shape == (2,)
    assert embeddings.shape == residuals.shape
    assert torch.equal(ids, torch.zeros(2, dtype=torch.long))
    assert torch.equal(embeddings, torch.zeros_like(residuals))
    assert batch_cluster_counts is None
    assert batch_cluster_sums is None
    assert layer.init_buffer.shape == (2, 2)
    assert not layer.is_initialized
    assert not hasattr(layer, "loss_function")
    assert not hasattr(layer, "cluster_counts")
    assert not hasattr(layer, "is_initial_step")
    assert not hasattr(layer, "init_centroids")
    assert not hasattr(layer, "initialize_on_cpu")


def test_quantization_models_do_not_expose_initialize_on_cpu():
    callables = [
        KMeansLayer,
        ResidualKMeans,
        ResidualVectorQuantization,
        ResidualQuantizationVAE,
        _kmeans_plus_plus_init,
    ]

    for callable_object in callables:
        assert "initialize_on_cpu" not in inspect.signature(callable_object).parameters


def test_quantization_model_configs_do_not_expose_initialize_on_cpu():
    config_paths = [
        PROJECT_ROOT / "configs/model/rkmeans_train.yaml",
        PROJECT_ROOT / "configs/model/rkmeans_inference.yaml",
        PROJECT_ROOT / "configs/model/rvq_train.yaml",
        PROJECT_ROOT / "configs/model/rqvae_train.yaml",
    ]

    for config_path in config_paths:
        assert "initialize_on_cpu" not in config_path.read_text(encoding="utf-8")


def test_quantization_models_do_not_expose_training_loop_function_or_manual_optimization():
    model_classes = [
        ResidualKMeans,
        ResidualVectorQuantization,
        ResidualQuantizationVAE,
    ]

    for model_class in model_classes:
        assert "training_loop_function" not in inspect.signature(model_class).parameters
        if model_class is ResidualKMeans:
            model = create_residual_kmeans(n_layers=1)
        elif model_class is ResidualVectorQuantization:
            model = create_residual_vector_quantization(n_layers=1)
        else:
            model = model_class(
                n_layers=1,
                n_clusters=2,
                n_features=2,
                training_model_config=create_training_model_config(),
                init_buffer_size=4,
            )
        assert not hasattr(model, "training_loop_function")
        assert model.automatic_optimization


def test_quantization_models_receive_grouped_training_model_config():
    model_classes = [
        ResidualKMeans,
        ResidualVectorQuantization,
        ResidualQuantizationVAE,
    ]

    for model_class in model_classes:
        parameters = inspect.signature(model_class).parameters
        assert "training_model_config" in parameters
        assert "training_components" not in parameters
        assert "loss_function" not in parameters
        assert "optimizer" not in parameters
        assert "scheduler" not in parameters
        assert "reconstruction_loss_function" not in parameters


def test_quantization_models_do_not_create_metric_attributes():
    models = [
        create_residual_kmeans(n_layers=2),
        create_residual_vector_quantization(n_layers=2),
        ResidualQuantizationVAE(
            n_layers=2,
            n_clusters=2,
            n_features=2,
            training_model_config=create_training_model_config(),
            init_buffer_size=4,
        ),
    ]
    metric_attribute_names = [
        "train_loss",
        "train_quantization_loss",
        "train_reconstruction_loss",
        "train_first_residuals_norm_ratio",
        "train_last_residuals_norm_ratio",
        "first_centroids_norm",
        "last_centroids_norm",
        "train_frac_unique_ids",
        "train_mse",
        "val_loss",
        "test_loss",
        "train_layer_coverages_0",
        "train_layer_id_entropy_0",
    ]

    for model in models:
        for attribute_name in metric_attribute_names:
            assert not hasattr(model, attribute_name)


def test_quantization_train_configs_declare_runtime_metrics_with_repeat():
    config_paths = [
        PROJECT_ROOT / "configs/model/rkmeans_train.yaml",
        PROJECT_ROOT / "configs/model/rvq_train.yaml",
        PROJECT_ROOT / "configs/model/rqvae_train.yaml",
    ]

    for config_path in config_paths:
        config = OmegaConf.load(config_path)
        config_container = OmegaConf.to_container(config, resolve=False)
        train_metrics = config_container["metrics"]["stages"]["train"]

        assert config.metrics._target_ == "src.common.metrics.MetricEngine"
        assert train_metrics["layer_coverages"]["repeat"]["count"] == "${num_hierarchies}"
        assert train_metrics["layer_coverages"]["repeat"]["spec"]["index"] == "{layer_idx}"
        assert train_metrics["layer_id_entropy"]["repeat"]["count"] == "${num_hierarchies}"
        assert train_metrics["layer_id_entropy"]["repeat"]["spec"]["index"] == "{layer_idx}"
        assert "optional" not in train_metrics["first_residuals_norm_ratio"]["spec"]
        assert "optional" not in train_metrics["layer_coverages"]["repeat"]["spec"]
        assert "loss" in config.metrics.stages.val
        assert "loss" in config.metrics.stages.test

    rqvae_config = OmegaConf.load(PROJECT_ROOT / "configs/model/rqvae_train.yaml")
    assert "reconstruction_loss" in rqvae_config.metrics.stages.train


def test_rkmeans_inference_config_matches_train_model_structure():
    train_config = OmegaConf.load(PROJECT_ROOT / "configs/model/rkmeans_train.yaml")
    inference_config = OmegaConf.load(PROJECT_ROOT / "configs/model/rkmeans_inference.yaml")
    train_root = OmegaConf.to_container(train_config.root, resolve=False)
    inference_root = OmegaConf.to_container(inference_config.root, resolve=False)

    assert inference_root["_target_"] == train_root["_target_"]
    assert inference_root["n_layers"] == train_root["n_layers"]
    assert inference_root["n_clusters"] == train_root["n_clusters"]
    assert inference_root["n_features"] == train_root["n_features"]
    assert inference_root["sub_layer"] == train_root["sub_layer"]
    assert "training_model_config" not in inference_config
    assert "training_model_config" not in inference_root
    assert "init_buffer_size" not in inference_root
    assert "optimizer" not in inference_root
    assert "scheduler" not in inference_root


def test_quantization_output_stats_return_metric_payload_dict():
    models = [
        create_residual_kmeans(n_layers=2),
        create_residual_vector_quantization(n_layers=2),
        ResidualQuantizationVAE(
            n_layers=2,
            n_clusters=2,
            n_features=2,
            training_model_config=create_training_model_config(),
            init_buffer_size=4,
        ),
    ]
    expected_keys = {
        "first_residuals_norm_ratio",
        "last_residuals_norm_ratio",
        "first_centroids_norm",
        "last_centroids_norm",
        "frac_unique_ids",
        "mse",
        "layer_coverages",
        "layer_id_entropies",
    }
    cluster_ids = torch.tensor([[0, 1], [1, 0]])
    all_residuals = torch.ones(2, 2, 2)
    input_embeddings = torch.ones(2, 2)

    for model in models:
        output_stats = model._compute_output_stats(
            cluster_ids=cluster_ids,
            all_residuals=all_residuals,
            input_embeddings=input_embeddings,
        )

        assert set(output_stats) == expected_keys
        assert len(output_stats["layer_coverages"]) == model.n_layers
        assert len(output_stats["layer_id_entropies"]) == model.n_layers


def test_quantization_training_configs_do_not_expose_training_loop_function():
    config_paths = [
        PROJECT_ROOT / "configs/model/rkmeans_train.yaml",
        PROJECT_ROOT / "configs/model/rkmeans_inference.yaml",
        PROJECT_ROOT / "configs/model/rvq_train.yaml",
        PROJECT_ROOT / "configs/model/rqvae_train.yaml",
    ]

    for config_path in config_paths:
        config_text = config_path.read_text(encoding="utf-8")
        assert "training_loop_function" not in config_text
        assert "scale_loss_by_world_size_for_initialization_training_loop" not in config_text


def test_rkmeans_does_not_expose_normalize_residuals():
    assert "normalize_residuals" not in inspect.signature(ResidualKMeans).parameters

    model = create_residual_kmeans(n_layers=1)

    assert not hasattr(model, "normalize_residuals")
    assert "normalize_residuals" not in (PROJECT_ROOT / "configs/model/rkmeans_inference.yaml").read_text(
        encoding="utf-8"
    )


def test_kmeans_layer_initializes_in_the_same_step_then_returns_batch_statistics():
    torch.manual_seed(0)
    layer = create_kmeans_layer(n_clusters=2, n_features=2, init_buffer_size=4)
    residuals = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )

    init_ids, init_embeddings, init_counts, init_sums = layer(residuals)

    assert init_ids.shape == (4,)
    assert init_embeddings.shape == residuals.shape
    assert init_counts is None
    assert init_sums is None
    assert layer.init_buffer.numel() == 0
    assert layer.is_initialized
    assert not hasattr(layer, "is_initial_step")
    assert not hasattr(layer, "init_centroids")
    assert not hasattr(layer, "initialize_on_cpu")
    assert not hasattr(layer, "loss_function")
    assert not hasattr(layer, "cluster_counts")
    assert not torch.equal(layer.centroids, torch.zeros_like(layer.centroids))

    ids, embeddings, batch_cluster_counts, batch_cluster_sums = layer(residuals)

    assert ids.shape == (4,)
    assert embeddings.shape == residuals.shape
    assert torch.sum(batch_cluster_counts).item() == residuals.shape[0]
    assert batch_cluster_sums.shape == (2, 2)
    assert layer.is_initialized


def test_kmeans_layer_distributed_rank_zero_broadcasts_initial_centroids(monkeypatch):
    torch.manual_seed(0)
    broadcast_calls = []
    layer = create_kmeans_layer(n_clusters=2, n_features=2, init_buffer_size=4)
    residuals = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])

    monkeypatch.setattr(distributed_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_utils.dist, "get_rank", lambda: 0)

    def fake_broadcast(tensor, src):
        broadcast_calls.append((tensor.detach().clone(), src))

    monkeypatch.setattr(distributed_utils.dist, "broadcast", fake_broadcast)

    layer(residuals)

    assert layer.is_initialized
    assert len(broadcast_calls) == 1
    assert broadcast_calls[0][1] == 0
    assert torch.equal(layer.centroids.detach(), broadcast_calls[0][0])


def test_kmeans_layer_distributed_non_zero_rank_receives_broadcasted_centroids(monkeypatch):
    broadcasted_centroids = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    layer = create_kmeans_layer(n_clusters=2, n_features=2, init_buffer_size=4)
    residuals = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])

    monkeypatch.setattr(distributed_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_utils.dist, "get_rank", lambda: 1)

    def fake_broadcast(tensor, src):
        assert src == 0
        tensor.copy_(broadcasted_centroids)

    monkeypatch.setattr(distributed_utils.dist, "broadcast", fake_broadcast)
    ids, embeddings, batch_cluster_counts, batch_cluster_sums = layer(residuals)

    assert batch_cluster_counts is None
    assert batch_cluster_sums is None
    assert layer.is_initialized
    assert torch.equal(layer.centroids.detach(), broadcasted_centroids)
    assert torch.equal(ids, torch.tensor([0, 0, 1, 1]))
    assert torch.equal(embeddings, broadcasted_centroids[ids])


def test_kmeans_layer_predict_does_not_update_runtime_state_or_parameters():
    layer = create_kmeans_layer(n_clusters=2, n_features=2, init_buffer_size=2)
    with torch.no_grad():
        layer.centroids.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    layer.is_initialized = True
    before_centroids = layer.centroids.detach().clone()

    ids, embeddings = layer.predict(torch.tensor([[0.8, 0.1], [0.2, 0.7]]))

    assert torch.equal(ids, torch.tensor([0, 1]))
    assert torch.equal(embeddings, before_centroids[ids])
    assert torch.equal(layer.centroids, before_centroids)


def test_residual_kmeans_uses_one_kmeans_layer_per_hierarchy():
    model = create_residual_kmeans(n_layers=3, init_buffer_size=2)

    assert len(model.layers) == 3
    assert all(isinstance(layer, KMeansLayer) for layer in model.layers)
    assert [tuple(layer.centroids.shape) for layer in model.layers] == [(2, 2), (2, 2), (2, 2)]
    assert "centroids_list" not in dict(model.named_parameters())
    assert "layers.0.centroids" in dict(model.named_parameters())
    assert len(model.cluster_counts_list) == 3
    assert all(not hasattr(layer, "cluster_counts") for layer in model.layers)


def test_rvq_uses_one_vector_quantization_layer_per_hierarchy():
    model = create_residual_vector_quantization(n_layers=3, init_buffer_size=2)

    assert len(model.layers) == 3
    assert all(isinstance(layer, VectorQuantizationLayer) for layer in model.layers)
    assert [tuple(layer.centroids.shape) for layer in model.layers] == [(2, 2), (2, 2), (2, 2)]
    assert "centroids_list" not in dict(model.named_parameters())
    assert "layers.0.centroids" in dict(model.named_parameters())
    assert not hasattr(model, "init_buffers")
    assert not hasattr(model, "is_initialized_list")
    assert all(not hasattr(layer, "loss_function") for layer in model.layers)


def test_residual_kmeans_forward_trains_only_current_layer_and_preserves_shapes():
    model = create_residual_kmeans(n_layers=2, init_buffer_size=2)
    model._trainer = SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.FITTING))
    model.current_layer = 1

    with torch.no_grad():
        model.layers[0].centroids.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.layers[1].centroids.copy_(torch.tensor([[0.5, 0.0], [0.0, 0.5]]))
    for layer in model.layers:
        layer.is_initialized = True

    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    cluster_ids, all_residuals, quantized_embeddings, quantization_loss = model.forward(embeddings)

    assert cluster_ids.shape == (3, 2)
    assert all_residuals.shape == (3, 2, 2)
    assert quantized_embeddings.shape == embeddings.shape
    assert quantization_loss.ndim == 0
    assert torch.sum(model.cluster_counts_list[0]).item() == 0
    assert torch.sum(model.cluster_counts_list[1]).item() == embeddings.shape[0]


def test_residual_kmeans_current_layer_initializes_with_single_state():
    torch.manual_seed(0)
    model = create_residual_kmeans(n_layers=2, init_buffer_size=3)
    model._trainer = SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.FITTING))
    model.current_layer = 1

    with torch.no_grad():
        model.layers[0].centroids.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    model.layers[0].is_initialized = True

    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    cluster_ids, all_residuals, quantized_embeddings, quantization_loss = model.forward(embeddings)

    assert model.layers[1].is_initialized
    assert not hasattr(model.layers[1], "is_initial_step")
    assert cluster_ids.shape == (3, 2)
    assert all_residuals.shape == (3, 2, 2)
    assert quantized_embeddings.shape == embeddings.shape
    assert quantization_loss.ndim == 0
    assert quantization_loss.requires_grad


def test_residual_kmeans_initialization_loss_keeps_grad_path_before_buffer_is_full():
    model = create_residual_kmeans(n_layers=1, init_buffer_size=4)
    model._trainer = SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.FITTING))
    model.current_layer = 0

    _, _, _, quantization_loss = model.forward(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    assert not model.layers[0].is_initialized
    assert quantization_loss.ndim == 0
    assert quantization_loss.requires_grad
    quantization_loss.backward()
    assert model.layers[0].centroids.grad is not None


def test_rvq_initializes_in_the_same_step_without_transition_state():
    torch.manual_seed(0)
    model = create_residual_vector_quantization(n_layers=1, init_buffer_size=4)
    residuals = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])

    ids, embeddings, quantization_loss_embeddings = model.layers[0](residuals)

    assert ids.shape == (4,)
    assert embeddings.shape == residuals.shape
    assert quantization_loss_embeddings is None
    assert model.layers[0].is_initialized
    assert model.layers[0].init_buffer.numel() == 0
    assert not hasattr(model, "is_initial_step_list")
    assert not hasattr(model, "init_centroids_list")
    assert not hasattr(model, "init_loss_function")
    assert not torch.equal(model.layers[0].centroids.detach(), torch.zeros_like(model.layers[0].centroids))


def test_rqvae_initializes_in_the_same_step_after_convergence_without_transition_state():
    torch.manual_seed(0)
    model = ResidualQuantizationVAE(
        n_layers=1,
        n_clusters=2,
        n_features=2,
        training_model_config=create_training_model_config(),
        init_buffer_size=4,
        kmeans_max_iter=5,
    )
    residuals = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])

    ids, embeddings, loss = model._layer_model_step(0, residuals)

    assert ids.shape == (4,)
    assert embeddings.shape == residuals.shape
    assert loss is not None
    assert model.is_initialized_list == [True]
    assert model.init_buffers[0].numel() == 0
    assert not hasattr(model, "is_initial_step_list")
    assert not hasattr(model, "init_centroids_list")
    assert not hasattr(model, "init_loss_function")
    assert not torch.equal(model.centroids_list[0].detach(), torch.zeros_like(model.centroids_list[0]))


def test_rvq_distributed_non_zero_rank_receives_broadcasted_initial_centroids(monkeypatch):
    broadcasted_centroids = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    model = create_residual_vector_quantization(n_layers=1, init_buffer_size=4)
    residuals = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])

    monkeypatch.setattr(distributed_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_utils.dist, "get_rank", lambda: 1)

    def fake_broadcast(tensor, src):
        assert src == 0
        tensor.copy_(broadcasted_centroids)

    monkeypatch.setattr(distributed_utils.dist, "broadcast", fake_broadcast)
    ids, embeddings, quantization_loss_embeddings = model.layers[0](residuals)

    assert quantization_loss_embeddings is None
    assert model.layers[0].is_initialized
    assert torch.equal(model.layers[0].centroids.detach(), broadcasted_centroids)
    assert torch.equal(ids, torch.tensor([0, 0, 1, 1]))
    assert torch.equal(embeddings, broadcasted_centroids[ids])


def test_rqvae_rank_zero_broadcasts_refined_initial_centroids(monkeypatch):
    torch.manual_seed(0)
    broadcast_calls = []
    model = ResidualQuantizationVAE(
        n_layers=1,
        n_clusters=2,
        n_features=2,
        training_model_config=create_training_model_config(),
        init_buffer_size=4,
        kmeans_max_iter=5,
    )
    residuals = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])

    monkeypatch.setattr(distributed_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_utils.dist, "get_rank", lambda: 0)

    def fake_broadcast(tensor, src):
        broadcast_calls.append((tensor.detach().clone(), src))

    monkeypatch.setattr(distributed_utils.dist, "broadcast", fake_broadcast)

    model._layer_model_step(0, residuals)

    assert model.is_initialized_list == [True]
    assert len(broadcast_calls) == 1
    assert broadcast_calls[0][1] == 0
    assert torch.equal(model.centroids_list[0].detach(), broadcast_calls[0][0])
