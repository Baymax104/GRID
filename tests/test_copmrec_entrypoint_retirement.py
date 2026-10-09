"""核验旧入口退出及基础流程保留，不执行实验。"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import MissingConfigException

import src.utils.hydra_resolvers  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/archive/copmrec-conversion-entrypoints-20261003"
MANIFEST = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))


def test_retired_files_are_archived_exactly_and_not_active():
    archive_path = ARCHIVE / MANIFEST["archive_file"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == MANIFEST["archive_sha256"]
    assert len(MANIFEST["files"]) == 23
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {entry["path"] for entry in MANIFEST["files"]}
        for entry in MANIFEST["files"]:
            payload = archive.read(entry["path"])
            assert len(payload) == entry["size"]
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
            assert not (ROOT / entry["path"]).exists()


@pytest.mark.parametrize("experiment", MANIFEST["retired_experiments"])
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
