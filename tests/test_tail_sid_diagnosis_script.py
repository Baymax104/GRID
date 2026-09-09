import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "tail_sid_diagnosis.sh"


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


def test_tail_sid_diagnosis_script_declares_required_passthrough_contract():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "--dry-run" in source
    assert "--notes=*" in source and "--notes)" in source
    assert "--group=*" in source and "--group)" in source
    assert 'group="$GROUP"' in source
    assert "--recommendation-output-path=*" in source
    assert "--recommendation-output-path)" in source
    assert 'ARGS+=("recommendation_output_path=$RECOMMENDATION_OUTPUT_PATH")' in source
    assert 'ARGS+=("${EXTRA_ARGS[@]}")' in source
    assert source.index('ARGS+=("${EXTRA_ARGS[@]}")') < source.index(
        'uv run --module src.main "${ARGS[@]}"'
    )


@pytest.mark.skipif(BASH is None, reason="bash is not available")
def test_tail_sid_diagnosis_script_has_valid_shell_syntax():
    result = subprocess.run([BASH, "-n", SCRIPT.as_posix()], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(BASH is None, reason="bash is not available")
def test_tail_sid_diagnosis_script_preserves_quoted_options_and_trailing_overrides(tmp_path):
    instrumented = tmp_path / "tail_sid_diagnosis.sh"
    source = SCRIPT.read_text(encoding="utf-8")
    source = source.replace('uv run --module src.main "${ARGS[@]}"', 'printf \'%s\\n\' "${ARGS[@]}"')
    instrumented.write_text(source, encoding="utf-8", newline="\n")

    result = subprocess.run(
        [
            BASH,
            instrumented.as_posix(),
            "--dry-run",
            "--notes",
            "diagnosis evidence run",
            "--group",
            "rqvae",
            "--semantic-id-path=wandb://entity/project/sid-run",
            "--recommendation-output-path",
            "wandb://rec-run",
            "tail_ratio=0.3",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert "semantic_id_path=wandb://entity/project/sid-run" in arguments
    assert "recommendation_output_path=wandb://rec-run" in arguments
    assert "group=rqvae" in arguments
    assert 'logger.wandb.notes="diagnosis evidence run"' in arguments
    assert "--dry-run" in arguments
    assert arguments[-1] == "tail_ratio=0.3"


@pytest.mark.skipif(BASH is None, reason="bash is not available")
def test_tail_sid_diagnosis_script_rejects_empty_notes_value():
    result = subprocess.run(
        [BASH, SCRIPT.as_posix(), "--notes", "--semantic-id-path=semantic.pt"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--notes requires a value" in result.stderr
