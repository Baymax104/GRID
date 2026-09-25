import shlex
import subprocess
from pathlib import Path

import pytest
from bash_fixture import bash as bash_fixture
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import src.utils.hydra_resolvers  # noqa: F401
from src.utils.launcher import apply_dry_run_overrides

bash = bash_fixture
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("control", ["legal_generation", "max_mixture"])
def test_mechanism_config(control):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=liger_joint_inference",
                f"model.root.mechanism_control={control}",
            ],
        )
    assert cfg.model.root.mechanism_control == control
    assert cfg.model.root.candidate_trace
    assert not cfg.model.root.content_only


def test_fixed_inference_alpha_config():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=liger_joint_inference",
                "model.root.inference_mixture_alpha=0.5",
            ],
        )
    assert cfg.model.root.inference_mixture_alpha == 0.5
    assert cfg.model.root.mechanism_control == "learned_mass"
    assert not cfg.model.root.content_only


def launch(bash, mode, extra=(), prefix=""):
    args = [
        "--data-dir",
        "data/beauty",
        "--dataset",
        "beauty",
        "--devices",
        "[0]",
        "--semantic-id-path",
        "wandb://baymaxam/GRID/dq77e3wo?role=semantic_id&file=merged_predictions_tensor.pt",
        "--embedding-path",
        "wandb://baymaxam/GRID/3jtt9mpa?role=semantic_embedding&file=merged_predictions_tensor.pt",
    ]
    if mode == "inference":
        args += ["--checkpoint", "wandb://baymaxam/GRID/future?role=checkpoint&file=a=b.ckpt"]
    command = 'uv() { printf "%s\\n" "$@"; }; export -f uv; ' + prefix
    command += "bash " + shlex.join([f"liger_joint_{mode}.sh", *args, *extra])
    return subprocess.run([bash], input=command, cwd=ROOT, text=True, capture_output=True)


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("equals", [False, True])
def test_joint_launcher_resolves(bash, mode, equals):
    note = 'literal $(no_execution) "quoted" \\ value'
    notes = ["--notes=" + note] if equals else ["--notes", note]
    output = launch(bash, mode, [*notes, "--dry-run", "seed=43"])
    assert output.returncode == 0, output.stderr
    argv = output.stdout.splitlines()
    assert argv[:3] == ["run", "-m", "src.main"]
    assert "--dry-run" in argv
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="main", overrides=[v for v in argv[3:] if v != "--dry-run"])
    cfg.paths.output_dir = "logs/test"
    cfg.paths.work_dir = "."
    cfg.paths.profile_dir = "logs/test/profile"
    OmegaConf.resolve(cfg)
    assert cfg.model.root._target_.endswith("JointMixtureLiger")
    assert cfg.seed == 43 and cfg.logger.wandb.notes == note
    assert cfg.group == "liger_joint_mixture_v1"
    assert cfg.model.root.candidate_strategy == "probability_mixture"
    assert cfg.model.root.mechanism_control == "learned_mass"
    assert cfg.model.root.inference_mixture_alpha is None
    if mode == "train":
        assert cfg.ckpt_path is None
        assert cfg.model.metrics.stages.train.mixture_loss.spec.key == "mixture_loss"
        assert cfg.trainer.root.max_steps == 50000
        assert cfg.model.training_model_config.scheduler.warmup_steps == 2500
        assert cfg.model.training_model_config.scheduler.scheduler_steps == 50000
        baseline_overrides = [
            "experiment=liger_train" if value == "experiment=liger_joint_train" else value
            for value in argv[3:]
            if value != "--dry-run"
        ]
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            baseline = compose(config_name="main", overrides=baseline_overrides)
        assert OmegaConf.to_container(cfg.model.training_model_config.optimizer) == OmegaConf.to_container(
            baseline.model.training_model_config.optimizer
        )
        for key in ["_target_", "_partial_", "min_ratio"]:
            assert cfg.model.training_model_config.scheduler[key] == baseline.model.training_model_config.scheduler[key]
        for key in baseline.trainer.root.keys():
            if key not in {"default_root_dir", "max_steps"}:
                assert cfg.trainer.root[key] == baseline.trainer.root[key]
        for loader in ["train_dataloader", "val_dataloader", "test_dataloader"]:
            assert cfg.data[loader].batch_size_per_device == baseline.data[loader].batch_size_per_device
    else:
        assert cfg.model.training_model_config is None
        assert cfg.model.root.candidate_trace
        assert cfg.data.datamodule.predict_dataloader_config.data_folder == "data/beauty/testing"
        assert cfg.ckpt_path.endswith("file=a=b.ckpt")
    cfg.dry_run = True
    apply_dry_run_overrides(cfg)
    assert cfg.logger.wandb is None
    if mode == "inference":
        assert cfg.callbacks.liger_trace_writer is None
        assert cfg.callbacks.wandb_artifact_writer is None
    else:
        assert cfg.trainer.root.max_steps == 1


@pytest.mark.parametrize("extra", [["--notes="], ["--notes"], ["--seed=-1"], ["--unknown"], ["--data-dir= "]])
def test_joint_rejects_bad_args(bash, extra):
    assert launch(bash, "train", extra).returncode == 2


def test_joint_shell_defaults_and_multi_gpu(bash):
    for mode in ["train", "inference"]:
        result = subprocess.run(
            [bash], input=f"bash -n liger_joint_{mode}.sh", cwd=ROOT, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr
    assert "--dry-run" not in launch(bash, "train").stdout
    multi = launch(bash, "train", prefix="export NPROC_PER_NODE=2; ")
    assert multi.returncode == 0
    assert multi.stdout.splitlines()[:3] == ["run", "torchrun", "--nproc_per_node=2"]
