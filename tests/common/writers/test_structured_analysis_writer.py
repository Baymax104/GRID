import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.writers import StructuredAnalysisOutput, StructuredAnalysisWriter


def _payload() -> StructuredAnalysisOutput:
    return StructuredAnalysisOutput(
        documents={"summary.json": {"schema_version": "v1", "verdict": {"risk": "supported"}}},
        tables={
            "group_metrics.csv": [{"group": "Tail", "risk": 0.5}],
            "item_damage_scores.csv": [{"item_id": 2, "raw_damage": 0.7}],
            "prefix_risk_scores.csv": [],
            "harmful_overlap_pairs.csv": [],
        },
        metadata={"schema_version": "v1", "task_name": "diagnosis"},
    )


def _run(writer: StructuredAnalysisWriter, payload: StructuredAnalysisOutput, trainer=None):
    writer.on_test_batch_end(
        trainer or SimpleNamespace(global_rank=0),
        SimpleNamespace(),
        {"structured_analysis": payload},
        batch=None,
        batch_idx=0,
    )


def test_structured_analysis_writer_writes_required_files_and_manifest_atomically(tmp_path):
    output_dir = tmp_path / "evidence"
    _run(StructuredAnalysisWriter(str(output_dir)), _payload())

    assert {path.name for path in output_dir.iterdir()} == {
        "summary.json",
        "group_metrics.csv",
        "item_damage_scores.csv",
        "prefix_risk_scores.csv",
        "harmful_overlap_pairs.csv",
        "manifest.json",
    }
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["complete"] is True
    with (output_dir / "group_metrics.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"group": "Tail", "risk": "0.5"}]


def test_structured_analysis_writer_does_not_expose_partial_output_on_serialization_failure(tmp_path):
    output_dir = tmp_path / "evidence"
    payload = StructuredAnalysisOutput(
        documents={"summary.json": {"bad": object()}}, tables={}, metadata={}
    )

    with pytest.raises(TypeError):
        _run(StructuredAnalysisWriter(str(output_dir)), payload)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".evidence-*"))


def test_structured_analysis_writer_publishes_directory_with_logger_owned_run(monkeypatch, tmp_path):
    artifacts = []

    class Artifact:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.directories = []
            artifacts.append(self)

        def add_dir(self, path):
            self.directories.append(Path(path))

    run = SimpleNamespace(log_artifact=lambda artifact, aliases: setattr(artifact, "aliases", aliases))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=Artifact))
    monkeypatch.setattr(
        "src.common.writers.structured_analysis_writer.require_wandb_logger_run",
        lambda trainer, purpose: run,
    )
    writer = StructuredAnalysisWriter(
        str(tmp_path / "evidence"),
        publish_wandb=True,
        artifact_name="evidence",
        artifact_type="diagnosis",
    )

    _run(writer, _payload())

    assert len(artifacts) == 1
    assert artifacts[0].directories == [tmp_path / "evidence"]
    assert artifacts[0].aliases == ["latest"]
    assert artifacts[0].kwargs["metadata"]["schema_version"] == "v1"


def test_structured_analysis_writer_propagates_publication_failure(monkeypatch, tmp_path):
    class Artifact:
        def __init__(self, **kwargs):
            pass

        def add_dir(self, path):
            pass

    def fail_publish(artifact, aliases):
        raise RuntimeError("publish failed")

    run = SimpleNamespace(log_artifact=fail_publish)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=Artifact))
    monkeypatch.setattr(
        "src.common.writers.structured_analysis_writer.require_wandb_logger_run",
        lambda trainer, purpose: run,
    )
    writer = StructuredAnalysisWriter(str(tmp_path / "evidence"), publish_wandb=True)

    with pytest.raises(RuntimeError, match="publish failed"):
        _run(writer, _payload())
