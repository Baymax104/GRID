from hydra import compose, initialize

import src.utils.hydra_resolvers  # noqa: F401


def _compose_experiment(experiment: str, extra_overrides: list[str] | None = None):
    with initialize(version_base=None, config_path="../configs"):
        return compose(
            config_name="main",
            overrides=[
                f"experiment={experiment}",
                "data_dir=data/beauty",
                "++embedding_path=wandb://01mw1fez",
                "++semantic_id_path=wandb://02sid",
                "ckpt_path=null",
                "++devices=[0,1]",
                *(extra_overrides or []),
            ],
        )


def _semantic_id_bundles(cfg):
    lists = []
    if "preprocessing_functions" in cfg.data:
        lists.append(cfg.data.preprocessing_functions)
    if "train_preprocessing_functions" in cfg.data:
        lists.append(cfg.data.train_preprocessing_functions)
    if "eval_preprocessing_functions" in cfg.data:
        lists.append(cfg.data.eval_preprocessing_functions)

    bundles = []
    for preprocessing_functions in lists:
        for preprocessing_function in preprocessing_functions:
            if "semantic_id_bundle" in preprocessing_function:
                bundles.append(preprocessing_function.semantic_id_bundle)
    return bundles


def test_wandb_loggers_use_experiment_user():
    for experiment in [
        "rkmeans_train",
        "rkmeans_inference",
        "rqvae_train",
        "rqvae_inference",
        "rvq_train",
        "rvq_inference",
        "sem_embeds_inference",
        "tail_sid_diagnosis",
        "tiger_train",
        "tiger_inference",
    ]:
        cfg = _compose_experiment(experiment)

        assert cfg.user == "baymaxam"
        assert cfg.logger.wandb.entity == cfg.user
        assert cfg.logger.wandb.project == cfg.project


def test_embedding_loader_configs_use_experiment_user():
    for experiment in [
        "rkmeans_train",
        "rkmeans_inference",
        "rqvae_train",
        "rqvae_inference",
        "rvq_train",
        "rvq_inference",
    ]:
        cfg = _compose_experiment(experiment)
        embedding_bundle = cfg.data.preprocessing_functions[-1].embedding_bundle

        assert embedding_bundle.wandb_entity == cfg.user
        assert embedding_bundle.wandb_project == cfg.project


def test_tiger_semantic_id_loader_configs_use_experiment_user():
    for experiment in ["tiger_train", "tiger_inference"]:
        cfg = _compose_experiment(experiment)

        assert cfg.model.root.semantic_ids.wandb_entity == cfg.user
        assert cfg.model.root.semantic_ids.wandb_project == cfg.project
        semantic_id_bundles = _semantic_id_bundles(cfg)
        assert semantic_id_bundles
        for semantic_id_bundle in semantic_id_bundles:
            assert semantic_id_bundle.wandb_entity == cfg.user
            assert semantic_id_bundle.wandb_project == cfg.project


def test_tiger_inference_uses_testing_last_item_holdout_preprocessing():
    cfg = _compose_experiment("tiger_inference")
    preprocessing_functions = cfg.data.preprocessing_functions
    targets = [step._target_ for step in preprocessing_functions]

    label_index = targets.index("src.data.components.preprocessing.generate_next_k_labels")
    normalize_index = targets.index("src.data.components.preprocessing.normalize_sequence")
    assert label_index < normalize_index

    label_step = preprocessing_functions[label_index]
    assert label_step.sequence_field_name == "sequence_data"
    assert label_step.input_field_name == "input_ids"
    assert label_step.target_field_name == "target_ids"
    assert label_step.next_k == cfg.model.root.num_hierarchies

    normalize_step = preprocessing_functions[normalize_index]
    assert normalize_step.input_field_name == "input_ids"
    assert normalize_step.sid_hierarchy == cfg.model.root.num_hierarchies
    assert cfg.data.collate.input_field_name == "input_ids"
    assert cfg.data.collate.target_field_name is None
    assert cfg.data.collate.output_key_field_name == "user_id"


def test_diagnosis_artifact_inputs_use_experiment_user():
    cfg = _compose_experiment("tail_sid_diagnosis")

    assert cfg.data.test_dataloader.semantic_id_path == "wandb://02sid"
    assert cfg.data.test_dataloader.embedding_path == "wandb://01mw1fez"
    assert cfg.data.test_dataloader.recommendation_output_path is None
    assert cfg.data.test_dataloader.wandb_entity == cfg.user
    assert cfg.data.test_dataloader.wandb_project == cfg.project

    cfg_without_embeddings = _compose_experiment("tail_sid_diagnosis", ["embedding_path=null"])

    assert cfg_without_embeddings.data.test_dataloader.embedding_path is None
    assert cfg_without_embeddings.data.test_dataloader.wandb_entity == cfg_without_embeddings.user
    assert cfg_without_embeddings.data.test_dataloader.wandb_project == cfg_without_embeddings.project

    cfg_with_recommendations = _compose_experiment(
        "tail_sid_diagnosis", ["recommendation_output_path=wandb://03rec"]
    )
    assert cfg_with_recommendations.data.test_dataloader.recommendation_output_path == "wandb://03rec"
