from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _python_sources(relative_dir: str):
    yield from (PROJECT_ROOT / relative_dir).rglob("*.py")


def test_common_and_data_do_not_reverse_import_tail_sid_diagnosis():
    forbidden = "src.quantization.tail_sid_diagnosis"
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for directory in ("src/common", "src/data")
        for path in _python_sources(directory)
        if forbidden in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_launcher_and_metric_callback_have_no_diagnosis_special_cases():
    paths = [
        PROJECT_ROOT / "src/utils/launcher.py",
        PROJECT_ROOT / "src/common/metrics/callback.py",
    ]

    assert all("tail_sid" not in path.read_text(encoding="utf-8").lower() for path in paths)


def test_only_lineage_callback_records_input_artifact_usage():
    callers = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in _python_sources("src")
        if ".use_artifact(" in path.read_text(encoding="utf-8")
    ]

    assert callers == ["src/common/callbacks/wandb_artifact_lineage.py"]
