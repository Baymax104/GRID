import shlex
import subprocess
from io import BytesIO
from pathlib import Path

import pytest
import torch
from bash_fixture import bash as bash_fixture
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import src.utils.hydra_resolvers  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
bash = bash_fixture


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_component_configs_follow_sasrec_protocol(mode):
    overrides = [
        f"experiment=sasrec_{mode}",
        "data_dir=data/beauty",
        "item_catalog_path=catalog.pt",
        "dataset_name=beauty",
    ]
    if mode == "inference":
        overrides.append("ckpt_path=best.ckpt")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="main", overrides=overrides)
    assert cfg.model.root._target_ == "src.recommendation.sasrec.SASRec"
    assert cfg.model.root.max_history_items == cfg.model.root.hidden_size == 50
    assert cfg.model.root.num_blocks == 2
    assert cfg.model.root.num_heads == 1
    assert cfg.model.root.dropout == 0.5
    assert cfg.model.training_model_config.optimizer._target_ == "torch.optim.Adam"
    assert list(cfg.model.training_model_config.optimizer.betas) == [0.9, 0.98]
    assert cfg.model.training_model_config.scheduler is None
    optimizer = instantiate(cfg.model.training_model_config.optimizer)([torch.nn.Parameter(torch.ones(1))])
    checkpoint = BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)
    assert torch.load(checkpoint, weights_only=True)["param_groups"][0]["betas"] == [0.9, 0.98]
    assert cfg.data.catalog.item_catalog_path == "catalog.pt"
    assert cfg.dry_run is False
    if mode == "train":
        assert cfg.trainer.root.max_steps == 50000
        assert cfg.data.train_dataloader.batch_size_per_device == 128
        assert cfg.data.val_dataloader.data_folder == "data/beauty/evaluation"
        assert cfg.callbacks.model_checkpoint.monitor == "val/ndcg@10"
        assert cfg.callbacks.model_checkpoint.mode == "max"
        assert cfg.run_test_after_training is False
        assert "test_dataloader" not in cfg.data
    else:
        assert cfg.data.predict_dataloader.data_folder == "data/beauty/testing"
        assert cfg.callbacks.wandb_artifact_writer.metadata.prediction_type == "item_ids"
        assert cfg.callbacks.prediction_metrics.logging_modes.test == "summary"


def invoke(bash, args, environment=""):
    script = 'uv() { printf "%s\\n" "$@"; }; export -f uv; unset NPROC_PER_NODE DEVICES MASTER_PORT WANDB_GROUP; '
    return subprocess.run(
        [bash],
        input=script + environment + " bash " + shlex.join(args),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def base_args(mode="train"):
    args = [
        f"sasrec_{mode}.sh",
        "--data-dir",
        "data/beauty",
        "--item-catalog-path",
        "catalog.pt",
        "--dataset-name",
        "beauty",
    ]
    if mode == "inference":
        args += ["--checkpoint", "best.ckpt"]
    return args


@pytest.mark.parametrize("equals", [False, True])
def test_script_quoting_notes_and_override_precedence(bash, equals):
    notes = 'official "SASRec" with spaces and backslash \\ test'
    args = base_args()
    args[2] = "data/beauty with spaces"
    args[4] = "wandb://entity/GRID/run?role=semantic_id&file=some file.pt"
    args += ["--notes=" + notes] if equals else ["--notes", notes]
    args += ["--dry-run", "model.root.dropout=0.2", "seed=200"]
    result = invoke(bash, args)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:5] == ["run", "python", "-m", "src.main", "experiment=sasrec_train"]
    assert 'data_dir="data/beauty with spaces"' in lines
    assert 'item_catalog_path="wandb://entity/GRID/run?role=semantic_id&file=some file.pt"' in lines
    assert 'logger.wandb.notes="official \\"SASRec\\" with spaces and backslash \\\\ test"' in lines
    assert "dry_run=true" in lines
    assert lines[-2:] == ["model.root.dropout=0.2", "seed=200"]


def test_empty_notes_and_default_real_training(bash):
    result = invoke(bash, base_args() + ["--notes", ""])
    assert result.returncode == 0, result.stderr
    assert "dry_run=false" in result.stdout.splitlines()
    assert "logger.wandb.notes=" not in result.stdout


def test_multi_gpu_uses_unified_torchrun_and_explicit_port(bash):
    result = invoke(bash, base_args("inference") + ["--devices=[0,1]", "--master-port", "29545"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:6] == [
        "run",
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=29545",
        "-m",
        "src.main",
    ]


@pytest.mark.parametrize(
    "extra",
    [
        ["--data-dir="],
        ["--seed", "0"],
        ["--seed=bad"],
        ["--devices=[]"],
        ["--devices=[-1]"],
        ["--master-port=65536"],
        ["--notes"],
        ["--unexpected"],
        ["garbage"],
    ],
)
def test_script_rejects_invalid_parameters(bash, extra):
    result = invoke(bash, base_args() + extra)
    assert result.returncode == 2


def test_script_rejects_inference_without_checkpoint_and_nproc_mismatch(bash):
    result = invoke(bash, ["sasrec_inference.sh", *base_args()[1:]])
    assert result.returncode == 2
    result = invoke(bash, base_args(), "export NPROC_PER_NODE=2;")
    assert result.returncode == 2


def test_shell_syntax(bash):
    for script in ("sasrec_common.sh", "sasrec_train.sh", "sasrec_inference.sh"):
        result = subprocess.run([bash, "-n", script], cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
