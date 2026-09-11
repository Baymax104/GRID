import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "tiger_prefix_trace.sh"


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


def _instrument(tmp_path: Path) -> Path:
    source = SCRIPT.read_text(encoding="utf-8")
    source = source[: source.index("OMP_NUM_THREADS=")] + "printf '%s\\n' \"${ARGS[@]}\"\n"
    path = tmp_path / SCRIPT.name
    path.write_text(source, encoding="utf-8", newline="\n")
    return path


def _required_args() -> list[str]:
    return [
        "--data-dir", "data/test dataset",
        "--data-split", "evaluation",
        "--beam-width", "10",
        "--devices", "[0]",
        "--group", "rkmeans",
        "--ckpt-path", "wandb://checkpoint",
        "--semantic-id-path", "wandb://sid",
        "--notes", "trace run with spaces",
    ]


def test_trace_script_declares_full_passthrough_contract():
    source = SCRIPT.read_text(encoding="utf-8")
    for option in [
        "--data-dir", "--data-split", "--beam-width", "--seed", "--master-port",
        "--devices", "--group", "--ckpt-path", "--semantic-id-path", "--notes", "--dry-run",
    ]:
        assert option in source
    assert 'ARGS+=("${EXTRA_ARGS[@]}")' in source


@pytest.mark.skipif(BASH is None, reason="bash is not available")
def test_trace_script_has_valid_shell_syntax():
    result = subprocess.run([BASH, "-n", SCRIPT.as_posix()], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(BASH is None, reason="bash is not available")
def test_trace_script_preserves_both_flag_forms_quotes_and_trailing_override(tmp_path):
    script = _instrument(tmp_path)
    result = subprocess.run(
        [
            BASH, script.as_posix(),
            "--data-dir=data/test dataset",
            "--data-split=evaluation",
            "--beam-width=50",
            "--seed", "17",
            "--master-port=29600",
            "--devices", "[1]",
            "--group=rvq",
            "--ckpt-path", "wandb://checkpoint",
            "--semantic-id-path=wandb://sid",
            "--notes", "quoted trace notes",
            "--dry-run",
            "beam_width=20",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert "experiment=tiger_prefix_trace" in arguments
    assert "data_dir=data/test dataset" in arguments
    assert "data_split=evaluation" in arguments
    assert "beam_width=50" in arguments
    assert 'logger.wandb.notes="quoted trace notes"' in arguments
    assert "--dry-run" in arguments
    assert arguments[-1] == "beam_width=20"


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--data-split=invalid"], "--data-split must be evaluation or testing"),
        (["--beam-width=0"], "--beam-width must be a positive integer"),
        (["--notes="], "--notes requires a value"),
    ],
)
def test_trace_script_rejects_invalid_or_empty_values(tmp_path, arguments, message):
    script = _instrument(tmp_path)
    base = _required_args()
    result = subprocess.run(
        [BASH, script.as_posix(), *base, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert message in result.stderr
