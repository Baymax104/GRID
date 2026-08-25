from hydra import compose, initialize

import src.utils.hydra_resolvers  # noqa: F401


def _compose_experiment(experiment: str):
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
