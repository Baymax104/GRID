import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.utils import instantiate

import src.utils.hydra_resolvers  # noqa: F401
from src.data.components.artifacts import load_item_resolution_calibration
from src.data.components.item_resolution import validate_resolution_trace
from src.recommendation.tiger_item_resolution.module import ARMS

ROOT = Path(__file__).resolve().parents[1]


def configuration(arm="mir", mode="train", extra=()):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="main",
            overrides=[
                f"experiment=tiger_item_resolution_{mode}",
                "data_dir=data/beauty",
                "dataset_name=beauty",
                "devices=[0]",
                "group=rkmeans",
                "semantic_id_path=sid.pt",
                "embedding_path=emb.pt",
                f"resolution_arm={arm}",
                *(["ckpt_path=checkpoint.ckpt"] if mode != "train" else []),
                *(["data_split=evaluation"] if mode == "inference" else []),
                *extra,
            ],
        )


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("arm", ARMS)
def test_all_conditions_compose_instantiate_and_save_contract(mode, arm, monkeypatch, tmp_path):
    cfg = configuration(
        arm,
        mode,
        [
            "model.root.embedding_dim=8",
            "model.root.projection_dim=4",
            "beam_width=4",
            "codebook_size=4",
            "model.encoder.config.num_heads=2",
            "model.encoder.config.num_layers=1",
            "model.encoder.config.d_ff=16",
            "model.encoder.config.d_kv=4",
        ],
    )
    sids = torch.tensor([[i // 8, i // 4 % 2, i // 2 % 2, i % 2] for i in range(16)])
    values = dict(
        keys=torch.arange(16),
        semantic_ids=sids,
        embeddings=torch.randn(16, 6, generator=torch.Generator().manual_seed(3)),
    )
    monkeypatch.setattr("src.data.components.artifacts.load_semantic_id_tensor", lambda **kw: sids)
    monkeypatch.setattr("src.data.components.catalog_content.load_catalog_content", lambda **kw: values)
    model = instantiate(cfg.model.root)
    assert model.arm == arm and model.warmup_steps == 2000
    assert cfg.experiment_protocol == "mir-v1"
    cfg.paths.output_dir = tmp_path.as_posix()
    if mode == "train":
        assert cfg.trainer.root.max_steps == 40000 and cfg.trainer.root.val_check_interval == 500
        assert cfg.trainer.root.strategy == "ddp_find_unused_parameters_true"
        assert cfg.callbacks.model_checkpoint.save_last is True
        assert cfg.callbacks.wandb_checkpoint_writer.role == "checkpoint"
        assert cfg.callbacks.wandb_last_checkpoint_writer.role == "checkpoint_last"
        assert cfg.run_test_after_training is False and cfg.ckpt_path is None
        assert cfg.data.val_dataloader.data_folder.endswith("/evaluation")
        assert instantiate(cfg.callbacks.model_checkpoint).monitor == "val/ndcg@10"
    else:
        assert cfg.trainer.root.num_nodes == 1 and cfg.trainer.root.devices == [0]
        assert cfg.data.collate.output_key_field_name == "user_id"
        assert cfg.model.metrics is None and model.trace_resolution is True
        writer = instantiate(cfg.callbacks.resolution_trace_writer)
        assert writer.validator.func is validate_resolution_trace


def test_calibration_uses_training_and_independent_writer():
    cfg = configuration("hybrid", "calibration")
    assert cfg.data_split == "training" and cfg.inference_policy == "calibrate"
    assert cfg.data.predict_dataloader.data_folder == "data/beauty/training"
    assert "recommendation_artifact_writer" not in cfg.callbacks
    assert cfg.callbacks.calibration_writer.role == "item_resolution_calibration"
    assert cfg.data.datamodule._target_.endswith("ItemResolutionDataModule")


def test_audit_config_instantiates_model_data_and_writer(monkeypatch, tmp_path):
    cfg = configuration(
        "mir",
        "audit",
        [
            "model.root.embedding_dim=8",
            "model.root.projection_dim=4",
            "beam_width=4",
            "codebook_size=4",
            "model.encoder.config.num_heads=2",
            "model.encoder.config.num_layers=1",
            "model.encoder.config.d_ff=16",
            "model.encoder.config.d_kv=4",
        ],
    )
    sids = torch.tensor([[i // 8, i // 4 % 2, i // 2 % 2, i % 2] for i in range(16)])
    values = dict(
        keys=torch.arange(16),
        semantic_ids=sids,
        embeddings=torch.randn(16, 6, generator=torch.Generator().manual_seed(3)),
    )
    monkeypatch.setattr("src.data.components.artifacts.load_semantic_id_tensor", lambda **kw: sids)
    monkeypatch.setattr(
        "src.data.components.artifacts.load_model_output", lambda **kw: dict(keys=values["keys"], predictions=sids)
    )
    monkeypatch.setattr("src.data.components.catalog_content.load_catalog_content", lambda **kw: values)
    model = instantiate(cfg.model.root)
    data = instantiate(cfg.data.datamodule)
    assert model.audit_state_budgets == (64, 128, 256) and model.audit_users == data.audit_users == 128
    assert cfg.run_mode == "inference" and cfg.data_split == "evaluation"
    assert cfg.data.predict_dataloader.timeout == 0 and cfg.data.predict_dataloader.num_workers == 0
    cfg.paths.output_dir = tmp_path.as_posix()
    writer = instantiate(cfg.callbacks.audit_writer)
    assert writer.payload_name == "item_resolution_audit"
    cfg.data.predict_dataloader.num_workers = 1
    with pytest.raises(Exception, match="num_workers=0"):
        instantiate(cfg.data.datamodule)


def test_audit_suite_prints_four_frozen_single_gpu_checkpoints(bash):
    result = invoke(
        bash,
        "audit_suite",
        [
            "--beauty-data-dir",
            "data/beauty",
            "--sports-data-dir",
            "data/sports",
            "--gpu",
            "3",
            "--notes",
            'paired "audit"',
            "--print-only",
            "audit_users=16",
        ],
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 4
    checkpoints = []
    for line in lines:
        tokens = shlex.split(line)
        assert tokens[0] == "CUDA_VISIBLE_DEVICES=3" and "torchrun" not in tokens
        start = next(i for i, token in enumerate(tokens) if token.startswith("experiment="))
        parsed = {o.key_or_group: o.value() for o in OverridesParser.create().parse_overrides(tokens[start:])}
        assert parsed["experiment"] == "tiger_item_resolution_audit" and parsed["audit_users"] == 16
        checkpoints.append(parsed["ckpt_path"])
    assert checkpoints == ["wandb://wgx7944n", "wandb://m47u9t4k", "wandb://y2ilymxo", "wandb://6jt7s9ro"]
    bad = invoke(bash, "audit", [*required("audit"), "--print-only", "--gpu=0,1"])
    assert bad.returncode != 0


def test_calibration_loader_validates_provenance(tmp_path):
    path = tmp_path / "calibration.pt"
    bundle = dict(
        schema_version="item_resolution_calibration_v1",
        keys=torch.tensor([3, 2]),
        labels=torch.zeros(2, 4),
        trace={"entropy": torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])},
        metadata=dict(source_split="training", checkpoint_fingerprint="checkpoint", catalog_fingerprint="catalog"),
    )
    torch.save(bundle, path)
    value = load_item_resolution_calibration(str(path))
    torch.testing.assert_close(value["thresholds"], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    bundle["metadata"]["source_split"] = "evaluation"
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="training split"):
        load_item_resolution_calibration(str(path))


@pytest.fixture
def bash():
    git = shutil.which("git")
    candidates = [shutil.which("bash"), str(Path(git).parents[1] / "bin/bash.exe") if git else None]
    for candidate in candidates:
        if candidate and Path(candidate).exists() and Path(candidate).parent.name.lower() != "system32":
            if subprocess.run([candidate, "-c", "exit 0"], capture_output=True).returncode == 0:
                return candidate
    pytest.skip("Usable Bash unavailable")


def required(mode):
    args = [
        "--data-dir",
        "data/test set",
        "--dataset",
        "beauty",
        "--semantic-id-path",
        "wandb://sid",
        "--embedding-path",
        "wandb://emb",
        "--notes",
        'paired "run", x=y; $(literal)',
    ]
    if mode != "train":
        args += ["--checkpoint-path", "wandb://checkpoint"]
    return args


def invoke(bash, mode, args):
    return subprocess.run(
        [bash, f"./tiger_item_resolution_{mode}.sh", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "NPROC_PER_NODE": "1"},
    )


@pytest.mark.parametrize("name", ["train", "inference", "calibration", "suite", "diagnosis", "audit", "audit_suite"])
def test_shell_syntax(bash, name):
    result = subprocess.run(
        [bash, "-n", f"./tiger_item_resolution_{name}.sh"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (
        subprocess.run(
            [bash, "-n", "./scripts/tiger_item_resolution_common.sh"], cwd=ROOT, capture_output=True
        ).returncode
        == 0
    )


@pytest.mark.parametrize("mode", ["train", "inference", "calibration", "audit"])
@pytest.mark.parametrize("equals", [False, True])
def test_arguments_quote_and_native_override(bash, mode, equals):
    args = required(mode)
    if equals:
        args = [f"{args[i]}={args[i + 1]}" for i in range(0, len(args), 2)]
    result = invoke(bash, mode, [*args, "--seed=17", "--dry-run", "--print-only", "seed=31"])
    assert result.returncode == 0, result.stderr
    tokens = shlex.split(result.stdout.strip())
    assert tokens[-1] == "seed=31" and "--dry-run" in tokens
    overrides = tokens[next(i for i, item in enumerate(tokens) if item.startswith("experiment=")) :]
    parsed = {
        item.key_or_group: item.value()
        for item in OverridesParser.create().parse_overrides([s for s in overrides if s != "--dry-run"])
    }
    assert parsed["data_dir"] == "data/test set"
    assert parsed["logger.wandb.notes"] == 'paired "run", x=y; $(literal)'
    if mode != "train":
        assert "torchrun" not in tokens and parsed["devices"] == [0]
        assert parsed["trainer.root.num_nodes"] == 1
    if mode == "calibration":
        assert parsed["resolution_arm"] == "hybrid" and parsed["data_split"] == "training"


@pytest.mark.parametrize(
    "bad",
    [["--data-dir="], ["--notes", " "], ["--seed=-1"], ["--master-port=65536"], ["--arm=unknown"], ["--unknown=1"]],
)
def test_invalid_args_rejected_before_launch(bash, bad):
    result = invoke(bash, "train", [*required("train"), "--print-only", *bad])
    assert result.returncode == 2 and "Error:" in result.stderr


@pytest.mark.parametrize(
    "bad",
    [
        ["--devices=[0,1]"],
        ["--gpu=0,1"],
        ["devices=[0,1]"],
        ["trainer.root.devices=2"],
        ["trainer.root.num_nodes=2"],
        ["--policy=wide"],
    ],
)
def test_inference_rejects_multigpu_and_missing_calibration(bash, bad):
    result = invoke(bash, "inference", [*required("inference"), "--print-only", *bad])
    assert result.returncode == 2, result.stdout


def test_complete_queues_are_disjoint_and_cover_54_conditions(bash):
    observed = []
    for queue in (1, 2):
        result = invoke(
            bash,
            "suite",
            [f"--queue={queue}", "--beauty-data-dir=data/beauty", "--sports-data-dir=data/sports", "--print-only"],
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 27
        for line in lines:
            tokens = shlex.split(line)
            assert tokens[0] == ("CUDA_VISIBLE_DEVICES=0,1" if queue == 1 else "CUDA_VISIBLE_DEVICES=2,3")
            assert "--nproc_per_node=2" in tokens
            parsed = {s.split("=", 1)[0]: s.split("=", 1)[1].strip('"') for s in tokens if "=" in s}
            observed.append((parsed["dataset_name"], parsed["resolution_arm"], int(parsed["seed"])))
    assert len(set(observed)) == 54
    assert set(observed) == {
        (dataset, arm, seed) for dataset in ("beauty", "sports") for arm in ARMS for seed in (42, 2024, 2025)
    }


@pytest.mark.parametrize(
    "bad", ["--seeds=42,42", "--arms=mir,mir", "--arms=invalid", "--seeds=42,", "resolution_arm=mir"]
)
def test_suite_rejects_ambiguous_matrix(bash, bad):
    result = invoke(bash, "suite", ["--queue=1", "--beauty-data-dir=a", "--sports-data-dir=b", "--print-only", bad])
    assert result.returncode == 2
