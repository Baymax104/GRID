from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.writers import source_snapshot


def test_code_artifact_uploads_files_not_local_references(tmp_path, monkeypatch):
    artifacts = []
    published = []

    class Artifact:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.files = []
            artifacts.append(self)

        def add_file(self, path, name):
            assert Path(path).is_file()
            self.files.append(name)

    class Logger:
        experiment = SimpleNamespace(id="test-run", log_artifact=published.append)

    import wandb

    monkeypatch.setattr(wandb, "Artifact", Artifact)
    monkeypatch.setattr(source_snapshot, "WandbLogger", Logger)
    for filename in ("source.tar.gz", "manifest.json", "runtime.json"):
        (tmp_path / filename).write_bytes(b"fixture")
    record = {
        "directory": str(tmp_path),
        "source_sha256": "source",
        "manifest_sha256": "manifest",
        "file_count": 10,
        "origin": {"status": "verified"},
    }
    source_snapshot.publish_source_snapshot([object(), Logger()], record)
    assert len(published) == 1
    assert artifacts[0].kwargs["type"] == "code"
    assert artifacts[0].kwargs["metadata"]["source_sha256"] == "source"
    assert artifacts[0].files == ["source.tar.gz", "manifest.json", "runtime.json"]


def test_non_wandb_logger_does_not_access_snapshot():
    source_snapshot.publish_source_snapshot([object()], {})


def test_nonzero_rank_never_accesses_experiment(monkeypatch):
    class Logger:
        @property
        def experiment(self):
            pytest.fail("Nonzero rank must not access W&B")

    monkeypatch.setattr(source_snapshot, "WandbLogger", Logger)
    monkeypatch.setattr(source_snapshot.rank_zero_only, "rank", 1)
    source_snapshot.publish_source_snapshot([Logger()], {})


def test_rank_zero_dummy_experiment_fails_closed(monkeypatch):
    from lightning.fabric.loggers.logger import _DummyExperiment

    class Logger:
        experiment = _DummyExperiment()

    monkeypatch.setattr(source_snapshot, "WandbLogger", Logger)
    monkeypatch.setattr(source_snapshot.rank_zero_only, "rank", 0)
    with pytest.raises(RuntimeError, match="requires a real W&B run"):
        source_snapshot.publish_source_snapshot([Logger()], {})


def test_missing_snapshot_file_fails_publication(tmp_path, monkeypatch):
    class Logger:
        experiment = SimpleNamespace(id="test-run")

    monkeypatch.setattr(source_snapshot, "WandbLogger", Logger)
    record = {
        "directory": str(tmp_path),
        "source_sha256": "source",
        "manifest_sha256": "manifest",
        "file_count": 10,
        "origin": {"status": "unavailable"},
    }
    with pytest.raises(ValueError, match="Path is not a file"):
        source_snapshot.publish_source_snapshot([Logger()], record)
