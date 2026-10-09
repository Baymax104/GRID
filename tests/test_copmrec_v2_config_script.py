"""脚本实际解析、Hydra 装配、单双卡及退休入口。"""

import shlex
import subprocess

import pytest
from bash_fixture import bash as bash_fixture
from formal_config_helpers import ROOT, compose_argv
from omegaconf import OmegaConf

bash = bash_fixture

BASE = [
    "--data-dir",
    "data/beauty spaced",
    "--dataset",
    "beauty",
    "--semantic-id-path",
    "wandb://fixture/GRID/sid?role=semantic_id&file=merged_predictions_tensor.pt",
    "--embedding-path",
    "content.pt",
    "--seed",
    "42",
]


def launch(bash, script, args, prefix=""):
    command = 'uv() { printf "%s\\n" "$@"; }; export -f uv; ' + prefix + "bash " + shlex.join([script, *BASE, *args])
    return subprocess.run([bash], input=command, cwd=ROOT, text=True, capture_output=True)


@pytest.mark.parametrize("variant", [None, "no_mixture", "no_residual", "no_joint_ce"])
@pytest.mark.parametrize("stage", ["train", "inference"])
@pytest.mark.parametrize("equals", [False, True])
def test_actual_script_composes_v2_single_or_dual_card_notes_and_overrides(bash, variant, stage, equals):
    notes = 'v2 "joint view", A=B'
    args = ["--devices", "[0,1]" if stage == "train" else "[0]"]
    args += ["--notes=" + notes] if equals else ["--notes", notes]
    if variant:
        args += ["--variant", variant]
    if stage == "inference":
        args += [
            "--checkpoint",
            "wandb://fixture/GRID/own?role=checkpoint&alias=v0&file=own.ckpt",
            "--checkpoint-sha256",
            "8" * 64,
        ]
    r = launch(bash, "copmrec_" + ("ablation_" if variant else "") + stage + ".sh", args)
    assert r.returncode == 0, r.stderr
    cfg = compose_argv(r.stdout.splitlines())
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert cfg.copmrec_version == cfg.implementation_version == cfg.method_version == "v2"
    assert cfg.logger.wandb.notes == notes
    assert cfg.native_view_ce is None and cfg.training_objective.native_view_computed is False
    assert not cfg.inference_history_exclusion
    assert ("--nproc_per_node=2" in r.stdout.splitlines()) == (stage == "train")
    if stage == "train":
        assert cfg.trainer.root.max_steps == 50000 and cfg.data.train_dataloader.batch_size_per_device == 128
    else:
        assert cfg.checkpoint_sha256 == "8" * 64 and cfg.data.predict_dataloader.data_folder.endswith("/testing")
    if variant:
        assert cfg.model.root.variant == variant and cfg.formal_release.variant == variant
        assert cfg.training_objective.joint_catalog_ce == int(variant != "no_joint_ce")
        assert cfg.training_objective.mixture_nll == int(variant != "no_mixture")


@pytest.mark.parametrize(
    "analysis,variant", [("hits", "full"), ("residual", "full"), ("prefix", "full"), ("prefix", "no_mixture")]
)
def test_diagnosis_uses_single_card_Trainer_test_and_v2(bash, analysis, variant):
    args = [
        "--analysis",
        analysis,
        "--variant",
        variant,
        "--devices",
        "[0]",
        "--dry-run",
        "--notes=quoted analysis",
        "logger.wandb.notes=override wins",
    ]
    if analysis != "hits":
        args += ["--checkpoint", "own.ckpt", "--checkpoint-sha256", "8" * 64]
    r = launch(bash, "copmrec_diagnosis.sh", args)
    assert r.returncode == 0, r.stderr
    cfg = compose_argv(r.stdout.splitlines())
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert cfg.run_mode == "analysis" and cfg.method_version == "v2"
    assert cfg.model.root.exclude_history is False and cfg.logger.wandb.notes == "override wins"
    assert "--dry-run" in r.stdout and "--nproc_per_node=2" not in r.stdout


@pytest.mark.parametrize(
    "script,args,prefix",
    [
        ("copmrec_train.sh", ["--devices", "[0,1]", "--checkpoint", "old.ckpt"], ""),
        ("copmrec_train.sh", ["--devices", "[0,1]", "--notes", ""], ""),
        ("copmrec_train.sh", ["--devices", "[0,1]"], "export NPROC_PER_NODE=1; "),
        ("copmrec_inference.sh", ["--devices", "[0]", "--checkpoint", "own.ckpt"], ""),
        ("copmrec_inference.sh", ["--devices", "[0]", "--checkpoint-sha256=bad"], ""),
        (
            "copmrec_inference.sh",
            ["--devices", "[0]", "--checkpoint", "own.ckpt", "--checkpoint-sha256", "8" * 64],
            "export NPROC_PER_NODE=2; ",
        ),
        ("copmrec_ablation_train.sh", ["--variant=no_native"], ""),
        ("copmrec_ablation_train.sh", ["--variant="], ""),
    ],
)
def test_reject_invalid_inputs_before_uv(bash, script, args, prefix):
    r = launch(bash, script, args, prefix)
    assert r.returncode == 2 and "src.main" not in r.stdout


def test_shell_syntax_and_only_current_experiments(bash):
    for script in ROOT.glob("copmrec*.sh"):
        r = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    assert {p.stem for p in (ROOT / "configs/experiment").glob("copmrec*.yaml")} == {
        "copmrec_train",
        "copmrec_inference",
        "copmrec_ablation_train",
        "copmrec_ablation_inference",
        "copmrec_diagnosis",
    }
