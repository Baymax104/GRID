import shutil
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = ["tiger_train.sh", "tiger_inference.sh", "tiger_prefix_trace.sh", "tail_sid_diagnosis.sh"]
BASE_ARGUMENTS = {
    "tiger_train.sh": [
        "--data-dir",
        "data/test dataset",
        "--notes",
        "grouped tiger train",
        "--semantic-id-path",
        "semantic.pt",
    ],
    "tiger_inference.sh": [
        "--data-dir",
        "data/test dataset",
        "--ckpt-path",
        "model.ckpt",
        "--semantic-id-path",
        "semantic.pt",
    ],
    "tiger_prefix_trace.sh": [
        "--data-dir",
        "data/test dataset",
        "--data-split",
        "evaluation",
        "--beam-width",
        "10",
        "--devices",
        "[0]",
        "--notes",
        "grouped trace",
        "--ckpt-path",
        "model.ckpt",
        "--semantic-id-path",
        "semantic.pt",
    ],
    "tail_sid_diagnosis.sh": [
        "--data-dir",
        "data/test dataset",
        "--notes",
        "grouped diagnosis",
        "--semantic-id-path",
        "semantic.pt",
    ],
}


def _usable_bash():
    candidates = [shutil.which("bash")]
    git = shutil.which("git")
    if git is not None:
        candidates.append(str(Path(git).parents[1] / "bin" / "bash.exe"))
    for candidate in candidates:
        if candidate is None or not Path(candidate).is_file():
            continue
        if Path(candidate).parent.name.lower() == "system32":
            continue
        result = subprocess.run(
            [candidate, "-c", "exit 0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return candidate
    return None


BASH = _usable_bash()


def _instrument_script(script_name: str, tmp_path: Path) -> Path:
    source = (PROJECT_ROOT / script_name).read_text(encoding="utf-8")
    print_args = "printf '%s\\n' \"${ARGS[@]}\"\n"
    if script_name == "tail_sid_diagnosis.sh":
        source = source.replace('uv run --module src.main "${ARGS[@]}"', print_args.rstrip())
    else:
        source = source[: source.index("OMP_NUM_THREADS=")] + print_args

    instrumented = tmp_path / script_name
    instrumented.write_text(source, encoding="utf-8", newline="\n")
    return instrumented


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_grouped_scripts_have_valid_shell_syntax(script_name):
    result = subprocess.run(
        [BASH, "-n", (PROJECT_ROOT / script_name).as_posix()],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
@pytest.mark.parametrize(
    ("group_arguments", "expected_group"),
    [
        (["--group=rkmeans"], "rkmeans"),
        (["--group", "rvq"], "rvq"),
        (["--group", "rqvae"], "rqvae"),
    ],
)
def test_grouped_scripts_forward_group_and_trailing_overrides(
    script_name, group_arguments, expected_group, tmp_path
):
    script = _instrument_script(script_name, tmp_path)
    result = subprocess.run(
        [
            BASH,
            script.as_posix(),
            *BASE_ARGUMENTS[script_name],
            *group_arguments,
            "trainer.root.limit_predict_batches=2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert f"group={expected_group}" in arguments
    assert arguments[-1] == "trainer.root.limit_predict_batches=2"


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_grouped_scripts_reject_missing_group(script_name, tmp_path):
    script = _instrument_script(script_name, tmp_path)
    result = subprocess.run(
        [BASH, script.as_posix(), *BASE_ARGUMENTS[script_name]],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--group requires one of: rkmeans, rvq, rqvae" in result.stderr


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_grouped_scripts_reject_unsupported_group(script_name, tmp_path):
    script = _instrument_script(script_name, tmp_path)
    result = subprocess.run(
        [BASH, script.as_posix(), *BASE_ARGUMENTS[script_name], "--group", "tiger"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unsupported --group 'tiger'" in result.stderr


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", ["tiger_train.sh", "tiger_inference.sh"])
@pytest.mark.parametrize("equals", [False, True])
def test_tiger_scripts_override_wandb_group(script_name, equals, tmp_path):
    script = _instrument_script(script_name, tmp_path)
    wandb_group = "paper_main_tiger_beauty"
    wandb_group_arguments = [f"--wandb-group={wandb_group}"] if equals else ["--wandb-group", wandb_group]
    result = subprocess.run(
        [
            BASH,
            script.as_posix(),
            *BASE_ARGUMENTS[script_name],
            "--group",
            "rkmeans",
            *wandb_group_arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert "group=rkmeans" in arguments
    assert f"group={wandb_group}" in arguments
    assert arguments.index(f"group={wandb_group}") > arguments.index("group=rkmeans")


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", ["tiger_train.sh", "tiger_inference.sh"])
def test_tiger_scripts_quote_wandb_uri_overrides_for_hydra(script_name, tmp_path):
    script = _instrument_script(script_name, tmp_path)
    semantic_id_uri = (
        "wandb://baymaxam/GRID/dq77e3wo"
        "?role=semantic_id&file=merged_predictions_tensor.pt"
    )
    arguments = [
        "--data-dir",
        "data/beauty",
        "--semantic-id-path",
        semantic_id_uri,
        "--group",
        "rkmeans",
        "--wandb-group",
        "paper_main_tiger_beauty",
    ]
    if script_name == "tiger_train.sh":
        arguments += ["--notes", "issue=BMX-34; TIGER; dataset=beauty; seed=42"]
    else:
        arguments += [
            "--ckpt-path",
            "wandb://baymaxam/GRID/train-run?role=checkpoint&file=best.ckpt",
        ]

    result = subprocess.run(
        [BASH, script.as_posix(), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    overrides = result.stdout.splitlines()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="main", overrides=overrides)
    assert cfg.semantic_id_path == semantic_id_uri
    assert cfg.logger.wandb.group == "paper_main_tiger_beauty"


@pytest.mark.skipif(BASH is None, reason="bash is not available")
def test_tiger_inference_quotes_notes_for_hydra(tmp_path):
    script = _instrument_script("tiger_inference.sh", tmp_path)
    notes = "issue=BMX-34; testing; TIGER; dataset=beauty; seed=42; metric-selected checkpoint"
    result = subprocess.run(
        [
            BASH,
            script.as_posix(),
            *BASE_ARGUMENTS["tiger_inference.sh"],
            "--group",
            "rkmeans",
            "--wandb-group",
            "paper_main_tiger_beauty",
            "--notes",
            notes,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    overrides = result.stdout.splitlines()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="main", overrides=overrides)
    assert cfg.logger.wandb.notes == notes
