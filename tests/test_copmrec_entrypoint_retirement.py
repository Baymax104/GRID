"""核验旧入口退出及基础流程保留，不执行实验。"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import MissingConfigException

import src.utils.hydra_resolvers  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
RETIRED_EXPERIMENTS = (
    "copmrec_decoder_train",
    "copmrec_decoder_audit",
    "copmrec_ranker_train",
    "copmrec_ranker_inference",
    "copmrec_scratch_train",
    "copmrec_scratch_train_ddp2",
    "copmrec_scratch_audit",
    "liger_joint_ranking_cache",
    "liger_joint_ranking_inference",
)
RETIRED_SCRIPTS = (
    "copmrec_decoder_common.sh",
    "copmrec_decoder_train.sh",
    "copmrec_decoder_audit.sh",
    "copmrec_ranker_common.sh",
    "copmrec_ranker_train.sh",
    "copmrec_ranker_inference.sh",
    "copmrec_scratch_train.sh",
    "copmrec_scratch_audit.sh",
    "liger_joint_ranking_cache.sh",
    "liger_joint_ranking_inference.sh",
)
RETIRED_CONTRACT_TESTS = (
    "test_copmrec_decoder_config_script.py",
    "test_copmrec_ranker_config_script.py",
    "test_copmrec_scratch_config_script.py",
    "test_copmrec_ranking_config_script.py",
)


def test_retired_entrypoint_files_are_not_active():
    retired_paths = (
        *RETIRED_SCRIPTS,
        *(f"configs/experiment/{name}.yaml" for name in RETIRED_EXPERIMENTS),
        *(f"tests/{name}" for name in RETIRED_CONTRACT_TESTS),
    )
    active_paths = [path for path in retired_paths if (ROOT / path).exists()]
    assert active_paths == []


@pytest.mark.parametrize("experiment", RETIRED_EXPERIMENTS)
def test_retired_experiment_cannot_compose(experiment):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        with pytest.raises(MissingConfigException, match=experiment):
            compose(config_name="main", overrides=[f"experiment={experiment}"])


@pytest.mark.parametrize(
    ("experiment", "target"),
    [
        ("liger_train", "src.recommendation.liger.Liger"),
        ("liger_inference", "src.recommendation.liger.Liger"),
    ],
)
def test_base_model_experiments_remain_available(experiment, target):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        config = compose(config_name="main", overrides=[f"experiment={experiment}"])
    assert config.model.root._target_ == target
    assert "foundation_steps" not in config.model.root
    assert "preservation_teacher_scope" not in config.model.root
