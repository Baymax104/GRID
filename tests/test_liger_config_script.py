import shlex
import subprocess
from pathlib import Path

import pytest
from bash_fixture import bash as bash_fixture
from hydra import compose, initialize_config_dir

import src.utils.hydra_resolvers  # noqa: F401
from src.utils.launcher import apply_dry_run_overrides

bash = bash_fixture
ROOT = Path(__file__).resolve().parents[1]


def launch(bash, mode, extra, env_prefix=""):
    args = [
        "--data-dir",
        "data/beauty",
        "--semantic-id-path",
        "wandb://baymaxam/GRID/sid?role=semantic_id&file=x=y.pt",
        "--embedding-path",
        "path with spaces/emb.pt",
        "--dataset",
        "beauty",
        "--devices",
        "[0]",
    ]
    if mode == "inference":
        args += ["--checkpoint", "wandb://baymaxam/GRID/ckpt?role=checkpoint&file=a=b.ckpt"]
    # 真正执行根脚本，替换进程边界而不复制其参数解析逻辑。
    command = 'uv() { printf "%s\\n" "$@"; }; export -f uv; ' + env_prefix
    command += "bash " + shlex.join([f"liger_{mode}.sh", *args, *extra])
    return subprocess.run([bash], input=command, cwd=ROOT, text=True, capture_output=True)


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("equals", [False, True])
def test_real_launcher_compose(bash, mode, equals):
    note = 'literal $(do_not_execute) "quoted" \\ text'
    notes = ["--notes=" + note] if equals else ["--notes", note]
    output = launch(bash, mode, [*notes, "--dry-run", "seed=43", "model.root.generation_candidates=40"])
    assert output.returncode == 0, output.stderr
    arguments = output.stdout.splitlines()
    assert arguments[:3] == ["run", "-m", "src.main"]
    overrides = [a for a in arguments[3:] if a != "--dry-run"]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="main", overrides=overrides)
    assert cfg.seed == 43 and cfg.logger.wandb.notes == note
    assert cfg.embedding_path == "path with spaces/emb.pt" and cfg.semantic_id_path.endswith("file=x=y.pt")
    assert cfg.model.root.generation_candidates == 40 and cfg.model.root.prediction_mode == "hybrid"
    assert cfg.model.root.catalog.training_data_dir == "data/beauty/training"
    assert cfg.sequence_length == 80
    if mode == "train":
        assert cfg.model.root.evaluation_mode == "dense"
        assert cfg.callbacks.model_checkpoint.monitor == "val/ndcg@10"
        assert cfg.trainer.root.max_steps == 200000
        assert cfg.model.training_model_config.scheduler.scheduler_steps == 200000
    else:
        assert cfg.model.training_model_config is None
        assert cfg.ckpt_path.endswith("file=a=b.ckpt")
    cfg.dry_run = True
    apply_dry_run_overrides(cfg)
    assert cfg.logger.wandb is None and not cfg.run_test_after_training
    if mode == "train":
        assert cfg.callbacks.model_checkpoint is None and cfg.trainer.root.max_steps == 1
    else:
        assert cfg.callbacks.wandb_artifact_writer is None


@pytest.mark.parametrize("extra", [["--notes="], ["--seed=-1"], ["--unknown"], ["--notes"], ["--embedding-path= "]])
def test_bad_args(bash, extra):
    assert launch(bash, "train", extra).returncode == 2


def test_no_default_dry_run_and_torchrun(bash):
    single = launch(bash, "train", [])
    assert "--dry-run" not in single.stdout
    multi = launch(bash, "train", [], "export NPROC_PER_NODE=2; ")
    assert multi.returncode == 0
    assert multi.stdout.splitlines()[:6] == [
        "run",
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=29730",
        "-m",
        "src.main",
    ]
    assert launch(bash, "train", [], "export MASTER_PORT=0; ").returncode == 2
    assert launch(bash, "train", [], "export NPROC_PER_NODE=0; ").returncode == 2


def test_shell_syntax(bash):
    for path in ["liger_train.sh", "liger_inference.sh", "scripts/liger_common.sh"]:
        result = subprocess.run([bash], input="bash -n " + shlex.quote(path), cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_candidate_trace_launcher_compose(bash):
    output = launch(
        bash,
        "inference",
        [
            "callbacks=liger_candidate_trace",
            "model.root.candidate_trace=true",
            "data.predict_dataloader.data_folder=data/beauty/evaluation",
        ],
    )
    assert output.returncode == 0, output.stderr
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="main", overrides=output.stdout.splitlines()[3:])
    assert cfg.model.root.candidate_trace
    assert cfg.data.datamodule.predict_dataloader_config.data_folder == "data/beauty/evaluation"
    assert cfg.callbacks.liger_trace_writer.role == "liger_candidate_trace"
    assert cfg.callbacks.wandb_artifact_writer.role == "recommendation_output"
    # 使用真实 dataloader 的 collate，避免手工构造带标签 batch 掩盖配置缺失。
    import torch
    from hydra.utils import instantiate

    collate = instantiate(cfg.data.datamodule.predict_dataloader_config.collate_fn)
    x, label = collate(
        [
            dict(
                input_ids=torch.tensor([0, 1, 2, 3]),
                attention_mask=torch.ones(4, dtype=torch.long),
                target_ids=torch.tensor([4, 5, 6, 7]),
                user_id=torch.tensor(123),
            )
        ]
    )
    assert label is not None
    assert label.target_ids.tolist() == [[4, 5, 6, 7]]
    assert x.input_ids.tolist() == [[0, 1, 2, 3]]
    assert x.output_keys.tolist() == [123]


@pytest.mark.parametrize("strategy,alpha", [("original", 0), ("probability_mixture", 0.5)])
def test_probability_mixture_launch_contract(bash, strategy, alpha):
    output = launch(
        bash,
        "inference",
        [
            "callbacks=liger_candidate_trace",
            "model.root.candidate_trace=true",
            f"model.root.candidate_strategy={strategy}",
            f"model.root.content_mixture_alpha={alpha}",
            "data.collate.target_field_name=target_ids",
            "data.predict_dataloader.data_folder=data/beauty/evaluation",
        ],
    )
    assert output.returncode == 0, output.stderr
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="main", overrides=output.stdout.splitlines()[3:])
    assert cfg.model.root.candidate_strategy == strategy
    assert cfg.model.root.content_mixture_alpha == alpha
    assert cfg.model.root.generation_candidates == 20 and cfg.model.root.top_k == 10
    assert cfg.data.datamodule.predict_dataloader_config.collate_fn.target_field_name == "target_ids"
    assert cfg.data.datamodule.predict_dataloader_config.data_folder == "data/beauty/evaluation"
