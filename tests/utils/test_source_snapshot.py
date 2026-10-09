import hashlib
import json
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.utils import launcher, source_snapshot


@pytest.fixture
def source_root(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for folder in ("src", "configs", "data", "logs", ".venv", "src/__pycache__"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for filename, content in {
        "src/main.py": "print('dirty')\n",
        "src/new.py": "# untracked\n",
        "src/__pycache__/cached.pyc": "cache",
        "configs/main.yaml": "seed: 42\n",
        "train.sh": "echo training\n",
        "sync.ps1": "Write-Output 'sync'\n",
        "pyproject.toml": "[project]\n",
        "uv.lock": "version = 1\n",
        ".env": "SECRET=must-not-be-archived",
        "data/item.py": "data",
        "logs/model.ckpt": "checkpoint",
        ".venv/runtime.py": "runtime",
    }.items():
        (root / filename).write_text(content, encoding="utf-8")
    monkeypatch.setattr(source_snapshot, "_runtime", lambda: {"python": "test", "packages": {}})
    return root


def test_archive_is_actual_bytes_and_excludes_assets(source_root, tmp_path):
    record = source_snapshot.create_source_snapshot(source_root, tmp_path / "run")
    directory = tmp_path / "run/metadata/source_snapshot"
    manifest = json.loads((directory / "manifest.json").read_text())
    with tarfile.open(directory / "source.tar.gz") as tar:
        names = set(tar.getnames())
        assert names == {
            "src/main.py",
            "src/new.py",
            "configs/main.yaml",
            "train.sh",
            "sync.ps1",
            "pyproject.toml",
            "uv.lock",
        }
        for entry in manifest["files"]:
            content = tar.extractfile(entry["path"]).read()
            assert content == (source_root / entry["path"]).read_bytes()
            assert hashlib.sha256(content).hexdigest() == entry["sha256"]
    assert record["source_sha256"] == manifest["source_sha256"]
    assert record["origin"]["status"] == "unavailable"


def test_local_git_origin_verified_then_mismatched_without_runtime_git(source_root, tmp_path, monkeypatch):
    subprocess.run(["git", "init", str(source_root)], check=True, capture_output=True)
    subprocess.run(["git", "add", "src/main.py"], cwd=source_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture"],
        cwd=source_root,
        check=True,
        capture_output=True,
    )
    (source_root / "src/main.py").write_text("# modified after commit\n")
    origin_file = source_snapshot.write_source_origin(source_root)
    origin = json.loads(origin_file.read_text())
    assert origin["git_dirty"] is True

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime must not query Git")

    monkeypatch.setattr(source_snapshot.subprocess, "run", forbidden)
    first = source_snapshot.create_source_snapshot(source_root, tmp_path / "first")
    assert first["origin"]["status"] == "verified"
    assert first["origin"]["git_commit"] == origin["git_commit"]
    (source_root / "src/new.py").write_text("# changed after sync\n")
    second = source_snapshot.create_source_snapshot(source_root, tmp_path / "second")
    assert second["origin"]["status"] == "mismatch"
    assert "git_commit" not in second["origin"]
    assert second["source_sha256"] != first["source_sha256"]


@pytest.mark.parametrize("origin", ["invalid JSON", "[]", '{"schema_version": 2}'])
def test_invalid_origin_does_not_claim_verified(source_root, tmp_path, origin):
    (source_root / source_snapshot.ORIGIN_FILE).write_text(origin)
    result = source_snapshot.create_source_snapshot(source_root, tmp_path / "run")
    assert result["origin"]["status"] == "invalid"


@pytest.mark.parametrize("dry_run,enabled,rank", [(True, True, "0"), (False, False, "0"), (False, True, "1")])
def test_dry_run_disabled_and_nonzero_rank_do_not_write(tmp_path, monkeypatch, dry_run, enabled, rank):
    monkeypatch.setenv("RANK", rank)
    cfg = OmegaConf.create({"dry_run": dry_run, "source_snapshot": {"enabled": enabled}})
    assert source_snapshot.prepare_source_snapshot(cfg) is None
    assert not list(tmp_path.iterdir())


def test_snapshot_failure_propagates(source_root, tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    cfg = OmegaConf.create(
        {
            "source_snapshot": {"enabled": True},
            "paths": {"work_dir": str(source_root), "output_dir": str(tmp_path / "run")},
        }
    )
    with pytest.raises(FileExistsError):
        source_snapshot.prepare_source_snapshot(cfg)
        source_snapshot.prepare_source_snapshot(cfg)


def test_lightning_subprocess_local_rank_skips_snapshot():
    import os
    import sys

    env = os.environ.copy()
    env.pop("RANK", None)
    env["LOCAL_RANK"] = "1"
    # 新进程验证 Lightning 在仅有 LOCAL_RANK 的启动环境中实际初始化的 rank。
    result = subprocess.run(
        [sys.executable, "-c", "\n".join([
            "from omegaconf import OmegaConf",
            "from lightning.pytorch.loggers import WandbLogger",
            "from lightning.fabric.loggers.logger import _DummyExperiment",
            "from src.utils.source_snapshot import prepare_source_snapshot",
            "from src.common.writers.source_snapshot import publish_source_snapshot",
            "logger = WandbLogger(project='rank-regression')",
            "assert isinstance(logger.experiment, _DummyExperiment)",
            "cfg = OmegaConf.create({'source_snapshot': {'enabled': True}})",
            "assert prepare_source_snapshot(cfg) is None",
            "publish_source_snapshot([logger], {})",
        ])],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_launcher_captures_and_publishes_before_execution(monkeypatch):
    calls = []
    cfg = OmegaConf.create({})
    snapshot = {"source_sha256": "fingerprint"}
    monkeypatch.setattr(launcher, "prepare_source_snapshot", lambda cfg: calls.append("capture") or snapshot)
    modules = SimpleNamespace(cfg=cfg, loggers=[], trainer=object())
    monkeypatch.setattr(launcher, "initialize_pipeline_modules", lambda cfg: calls.append("initialize") or modules)
    monkeypatch.setattr(launcher, "publish_source_snapshot", lambda *args: calls.append("publish"))
    monkeypatch.setattr(launcher, "log_hyperparameters", lambda *args: calls.append("config"))
    monkeypatch.setattr(launcher, "finalize_loggers", lambda *args: calls.append("finalize"))
    with launcher.pipeline_launcher(cfg):
        calls.append("execute")
    assert calls == ["capture", "initialize", "publish", "config", "execute", "finalize"]
    assert cfg.source_snapshot_record.source_sha256 == "fingerprint"


def test_publish_failure_prevents_task_and_finalizes(monkeypatch):
    cfg = OmegaConf.create({})
    calls = []
    monkeypatch.setattr(launcher, "prepare_source_snapshot", lambda cfg: {})
    monkeypatch.setattr(launcher, "initialize_pipeline_modules", lambda cfg: SimpleNamespace(loggers=[], trainer=None))
    monkeypatch.setattr(launcher, "finalize_loggers", lambda trainer: calls.append("finalize"))

    def fail(*args):
        raise RuntimeError("publication failed")

    monkeypatch.setattr(launcher, "publish_source_snapshot", fail)
    with pytest.raises(RuntimeError, match="publication failed"), launcher.pipeline_launcher(cfg):
        pytest.fail("Task must not execute")
    assert calls == ["finalize"]


@pytest.mark.parametrize("experiment", ["sasrec_train", "sasrec_inference", "tiger_train", "liger_train"])
def test_source_snapshot_config_enabled_by_default(experiment):
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="main", overrides=[f"experiment={experiment}"])
    assert cfg.source_snapshot.enabled is True
