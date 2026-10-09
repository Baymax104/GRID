import shlex
import subprocess
from pathlib import Path

import pytest
from bash_fixture import bash as bash_fixture
from hydra import compose, initialize_config_dir

import src.utils.hydra_resolvers  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
bash = bash_fixture


def test_tiger_paper_training_protocol():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=tiger_train",
                "data_dir=data/beauty",
                "semantic_id_path=semantic-id.pt",
                "devices=[0,1]",
                "group=rkmeans",
                "seed=42",
            ],
        )

    assert cfg.trainer.root.max_steps == 50000
    assert cfg.trainer.root.strategy == "ddp"
    assert cfg.trainer.root.precision == "32-true"
    assert cfg.trainer.root.accumulate_grad_batches == 1
    assert cfg.data.train_dataloader.batch_size_per_device == 128
    assert cfg.callbacks.model_checkpoint.monitor == "val/ndcg@10"
    assert cfg.callbacks.model_checkpoint.mode == "max"
    assert cfg.run_test_after_training is False


@pytest.mark.parametrize("equals", [False, True])
def test_tiger_train_master_port_cli_overrides_environment(bash, equals):
    port_option = ["--master-port=29545"] if equals else ["--master-port", "29545"]
    args = [
        "tiger_train.sh",
        "--data-dir",
        "data/beauty",
        "--semantic-id-path",
        "wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt",
        "--devices",
        "[0,1]",
        "--group",
        "rkmeans",
        "--seed",
        "42",
        "--notes",
        "master port contract",
        *port_option,
    ]
    command = (
        'uv() { printf "%s\\n" "$@"; }; export -f uv; '
        "export NPROC_PER_NODE=2; export MASTER_PORT=29544; bash "
        + shlex.join(args)
    )
    result = subprocess.run([bash], input=command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:6] == [
        "run",
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=29545",
        "-m",
        "src.main",
    ]


def test_tiger_train_rejects_invalid_master_port(bash):
    command = (
        'uv() { printf "%s\\n" "$@"; }; export -f uv; '
        "bash tiger_train.sh --data-dir data/beauty --semantic-id-path sid.pt "
        "--devices '[0,1]' --group rkmeans --seed 42 --notes test --master-port 0"
    )
    result = subprocess.run([bash], input=command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 2


def test_tiger_train_master_port_environment_override(bash):
    command = (
        'uv() { printf "%s\\n" "$@"; }; export -f uv; '
        "export NPROC_PER_NODE=2; export MASTER_PORT=29546; "
        "bash tiger_train.sh --data-dir data/beauty --semantic-id-path sid.pt "
        "--devices '[0,1]' --group rkmeans --seed 42 --notes test"
    )
    result = subprocess.run([bash], input=command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--master_port=29546" in result.stdout.splitlines()
