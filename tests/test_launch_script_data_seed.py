import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = [
    "sem_embeds_inference.sh",
    "rkmeans_train.sh",
    "rkmeans_inference.sh",
    "rvq_train.sh",
    "rvq_inference.sh",
    "rqvae_train.sh",
    "rqvae_inference.sh",
    "tiger_train.sh",
    "tiger_inference.sh",
    "tiger_prefix_trace.sh",
    "tail_sid_diagnosis.sh",
]
REQUIRED_ARGUMENTS = {
    "sem_embeds_inference.sh": [],
    "rkmeans_train.sh": ["--embedding-path", "embeddings.pt", "--notes", "train"],
    "rkmeans_inference.sh": [
        "--embedding-path",
        "embeddings.pt",
        "--ckpt-path",
        "model.ckpt",
    ],
    "rvq_train.sh": ["--embedding-path", "embeddings.pt", "--notes", "train"],
    "rvq_inference.sh": [
        "--embedding-path",
        "embeddings.pt",
        "--ckpt-path",
        "model.ckpt",
    ],
    "rqvae_train.sh": ["--embedding-path", "embeddings.pt", "--notes", "train"],
    "rqvae_inference.sh": [
        "--embedding-path",
        "embeddings.pt",
        "--ckpt-path",
        "model.ckpt",
    ],
    "tiger_train.sh": [
        "--semantic-id-path",
        "semantic.pt",
        "--notes",
        "train",
        "--group",
        "rvq",
    ],
    "tiger_inference.sh": [
        "--semantic-id-path",
        "semantic.pt",
        "--ckpt-path",
        "model.ckpt",
        "--group",
        "rvq",
    ],
    "tiger_prefix_trace.sh": [
        "--data-split",
        "evaluation",
        "--beam-width",
        "10",
        "--devices",
        "[0]",
        "--semantic-id-path",
        "semantic.pt",
        "--ckpt-path",
        "model.ckpt",
        "--notes",
        "trace",
        "--group",
        "rvq",
    ],
    "tail_sid_diagnosis.sh": [
        "--semantic-id-path",
        "semantic.pt",
        "--notes",
        "diagnosis",
        "--group",
        "rvq",
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


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", ["rkmeans_train.sh", "rvq_train.sh"])
@pytest.mark.parametrize("equals_form", [False, True])
@pytest.mark.parametrize(
    "embedding",
    [
        "wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt",
        'data/embedding = "quoted".pt',
        "/data/embedding with spaces.pt",
    ],
)
def test_quantizer_embedding_survives_shell_and_hydra(script_name, equals_form, embedding, tmp_path):
    script = _instrument_script(script_name, tmp_path)
    option = [f"--embedding-path={embedding}"] if equals_form else ["--embedding-path", embedding]
    command = [
        "bash",
        script.as_posix(),
        "--data-dir",
        "data/beauty",
        "--notes",
        "URI regression",
        *option,
        'embedding_path="override=last.pt"',
    ]
    result = subprocess.run(
        [BASH],
        input=shlex.join(command) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    parsed = OverridesParser.create().parse_overrides(result.stdout.splitlines())
    values = [item.value() for item in parsed if item.key_or_group == "embedding_path"]
    assert values == [embedding, "override=last.pt"]


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
def test_launch_scripts_require_data_dir(script_name, tmp_path):
    script = _instrument_script(script_name, tmp_path)
    result = subprocess.run(
        [BASH, script.as_posix(), *REQUIRED_ARGUMENTS[script_name]],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--data-dir requires a value" in result.stderr


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
@pytest.mark.parametrize(
    ("data_arguments", "seed_arguments", "expected_seed"),
    [
        (["--data-dir=data/test dataset"], [], "42"),
        (["--data-dir", "data/test dataset"], ["--seed=17"], "17"),
        (["--data-dir", "data/test dataset"], ["--seed", "23"], "23"),
    ],
)
def test_launch_scripts_forward_data_dir_and_seed(script_name, data_arguments, seed_arguments, expected_seed, tmp_path):
    script = _instrument_script(script_name, tmp_path)
    trailing_override = "seed=99"
    result = subprocess.run(
        [
            BASH,
            script.as_posix(),
            *REQUIRED_ARGUMENTS[script_name],
            *data_arguments,
            *seed_arguments,
            trailing_override,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert "data_dir=data/test dataset" in arguments
    assert f"seed={expected_seed}" in arguments
    assert arguments[-1] == trailing_override


@pytest.mark.skipif(BASH is None, reason="bash is not available")
@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
@pytest.mark.parametrize("option", ["--data-dir=", "--seed="])
def test_launch_scripts_reject_empty_data_dir_or_seed(script_name, option, tmp_path):
    script = _instrument_script(script_name, tmp_path)
    arguments = [*REQUIRED_ARGUMENTS[script_name], "--data-dir=data/test", option]
    result = subprocess.run(
        [BASH, script.as_posix(), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    expected_option = option.removesuffix("=")
    assert f"{expected_option} requires a value" in result.stderr
