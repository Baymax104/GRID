import shlex
import subprocess
from pathlib import Path

import pytest
from bash_fixture import bash as bash_fixture
from hydra import compose, initialize_config_dir

import src.utils.hydra_resolvers  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
bash = bash_fixture


@pytest.mark.parametrize("stage", ["tokenizer_train", "sid", "train", "inference", "cf_train", "cf_export"])
def test_letter_compose(stage):
    overrides = [f"experiment=letter_{stage}", "dataset_name=beauty"]
    if stage in ("cf_train", "cf_export"):
        overrides += ["embedding_path=content.pt", "data_dir=data/beauty"]
    elif stage in ("train", "inference"):
        overrides += ["semantic_id_path=sid.pt", "data_dir=data/beauty"]
    else:
        overrides += ["embedding_path=content.pt", "cf_embedding_path=cf.pt", "cf_source=train-only-source"]
    if stage in ("sid", "inference", "cf_export"):
        overrides.append("ckpt_path=best.ckpt")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="main", overrides=overrides)
    assert ".letter." in cfg.model.root._target_
    assert cfg.trainer.root.precision == "32-true" and not cfg.dry_run
    if stage == "train":
        assert cfg.trainer.root.max_steps == 50000 and cfg.trainer.root.val_check_interval == 500
        assert cfg.max_history_items == 20
        assert cfg.model.training_model_config.optimizer.lr == 0.0005
        assert cfg.callbacks.model_checkpoint.monitor == "val/ndcg@10"
        assert cfg.data.train_dataloader.batch_size_per_device == 128
    if stage == "inference":
        assert cfg.data.predict_dataloader.data_folder == "data/beauty/testing"
        assert cfg.callbacks.prediction_metrics.logging_modes.test == "summary"
    if stage == "cf_train":
        assert cfg.model.root._target_.endswith("LetterCFTeacher")
        assert cfg.model.root.max_history_items == 50
        assert cfg.trainer.root.val_check_interval == 1000
        assert cfg.model.training_model_config.optimizer.betas == [0.9, 0.98]
    if stage == "cf_export":
        assert cfg.callbacks.wandb_artifact_writer.role == "collaborative_embedding"


def test_cf_export_script_requires_single_process_and_checkpoint(bash):
    args = ["letter_cf_export.sh", "--dataset", "beauty", "--data-dir", "data/beauty", "--embedding-path", "content.pt"]
    assert invoke(bash, args).returncode == 2
    args += ["--ckpt-path", "best_step=001.ckpt", "--notes=CF source", "--dry-run", "seed=200"]
    result = invoke(bash, args)
    assert result.returncode == 0 and 'ckpt_path="best_step=001.ckpt"' in result.stdout
    assert 'logger.wandb.notes="CF source"' in result.stdout and result.stdout.splitlines()[-1] == "seed=200"
    assert invoke(bash, args + ["--gpus", "0,1", "--nproc-per-node", "2"]).returncode == 2


def invoke(bash, args):
    return subprocess.run(
        [bash],
        input='uv() { printf "%s\\n" "$@"; }; export -f uv; bash ' + shlex.join(args),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def base_args():
    return [
        "letter_train.sh",
        "--dataset",
        "beauty",
        "--data-dir",
        "data/beauty with spaces",
        "--semantic-id-path",
        "sid.pt",
    ]


@pytest.mark.parametrize("equals", [True, False])
def test_notes_and_override_precedence(bash, equals):
    notes = 'test "LETTER" \\ check'
    args = base_args() + (["--notes=" + notes] if equals else ["--notes", notes]) + ["--dry-run", "seed=200"]
    result = invoke(bash, args)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:6] == ["run", "torchrun", "--nproc_per_node=2", "--master_port=29521", "-m", "src.main"]
    assert 'data_dir="data/beauty with spaces"' in lines
    assert 'logger.wandb.notes="test \\"LETTER\\" \\\\ check"' in lines
    assert "dry_run=true" in lines and lines[-1] == "seed=200"


def test_empty_notes(bash):
    result = invoke(bash, base_args() + ["--notes", ""])
    assert result.returncode == 0 and "logger.wandb.notes=" not in result.stdout and "dry_run=false" in result.stdout


@pytest.mark.parametrize(
    "extra",
    [
        ["--notes"],
        ["--gpus", ""],
        ["--nproc-per-node", "3"],
        ["--master-port=65536"],
        ["--unknown"],
        ["garbage"],
        ["--seed=wrong"],
    ],
)
def test_invalid_script_args(bash, extra):
    assert invoke(bash, base_args() + extra).returncode == 2


def test_bash_syntax(bash):
    for path in ROOT.glob("letter_*.sh"):
        assert subprocess.run([bash, "-n", str(path)], capture_output=True).returncode == 0


def test_equal_sign_checkpoint_is_quoted(bash):
    args = base_args()
    args[0] = "letter_inference.sh"
    result = invoke(bash, args + ["--ckpt-path", "checkpoints/step_step=000005.ckpt"])
    assert result.returncode == 0
    assert 'ckpt_path="checkpoints/step_step=000005.ckpt"' in result.stdout.splitlines()
