import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = ["rkmeans_train.sh", "rvq_train.sh", "rqvae_train.sh"]
INFERENCE_SCRIPTS = [
    "rkmeans_inference.sh",
    "rvq_inference.sh",
    "rqvae_inference.sh",
]
SCRIPT_NAMES = TRAIN_SCRIPTS + INFERENCE_SCRIPTS


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


def _base_arguments(script_name: str) -> list[str]:
    if script_name in TRAIN_SCRIPTS:
        return ["--data-dir", "data/test dataset", "--notes", "explicit embedding source"]
    return ["--data-dir", "data/test dataset", "--ckpt-path", "wandb://checkpoint123"]


def _instrument_script(script_name: str, tmp_path: Path) -> Path:
    source = (PROJECT_ROOT / script_name).read_text(encoding="utf-8")
    source = source[: source.index("OMP_NUM_THREADS=")]
    source += "printf '%s\\n' \"${ARGS[@]}\"\n"

    instrumented = tmp_path / script_name
    instrumented.write_text(source, encoding="utf-8", newline="\n")
    return instrumented


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_quantization_scripts_have_valid_shell_syntax(script_name):
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
    ("embedding_arguments", "expected_embedding"),
    [
        (["--embedding-path=wandb://producer123"], "wandb://producer123"),
        (["--embedding-path", "outputs/semantic embeddings.pt"], "outputs/semantic embeddings.pt"),
    ],
)
def test_quantization_scripts_forward_embedding_and_trailing_overrides(
    script_name, embedding_arguments, expected_embedding, tmp_path
):
    script = _instrument_script(script_name, tmp_path)
    trailing_override = "embedding_path=wandb://override456"
    result = subprocess.run(
        [
            BASH,
            script.as_posix(),
            *_base_arguments(script_name),
            *embedding_arguments,
            "--dry-run",
            trailing_override,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert f"embedding_path={expected_embedding}" in arguments
    assert "--dry-run" in arguments
    assert arguments[-1] == trailing_override
    if script_name in TRAIN_SCRIPTS:
        assert 'logger.wandb.notes="explicit embedding source"' in arguments
    else:
        assert "ckpt_path=wandb://checkpoint123" in arguments


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_quantization_scripts_reject_missing_embedding(script_name, tmp_path):
    script = _instrument_script(script_name, tmp_path)
    result = subprocess.run(
        [BASH, script.as_posix(), *_base_arguments(script_name)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--embedding-path requires a local path or wandb://<run-id> value" in result.stderr


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
@pytest.mark.parametrize("embedding_arguments", [["--embedding-path="], ["--embedding-path", "--dry-run"]])
def test_quantization_scripts_reject_empty_embedding(
    script_name, embedding_arguments, tmp_path
):
    script = _instrument_script(script_name, tmp_path)
    result = subprocess.run(
        [
            BASH,
            script.as_posix(),
            *_base_arguments(script_name),
            *embedding_arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--embedding-path requires a local path or wandb://<run-id> value" in result.stderr


@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_quantization_scripts_do_not_pin_embedding_producer(script_name):
    source = (PROJECT_ROOT / script_name).read_text(encoding="utf-8")

    assert "embedding_path=wandb://vb8es5ow" not in source
    assert 'embedding_path="$EMBEDDING_PATH"' in source
