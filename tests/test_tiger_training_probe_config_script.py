import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.utils import instantiate

import src.utils.hydra_resolvers  # noqa: F401
from src.recommendation.tiger.tiger import Tiger

ROOT = Path(__file__).resolve().parents[1]


def configuration(arm="ce", extra=()):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="main",
            overrides=[
                "experiment=tiger_training_probe",
                "data_dir=data/beauty",
                "devices=[0]",
                "group=rkmeans",
                "semantic_id_path=sid.pt",
                "initialization_checkpoint_path=initial.ckpt",
                f"probe_arm={arm}",
                *extra,
            ],
        )


@pytest.mark.parametrize("arm", ["ce", "reweighted"])
def test_probe_composition_reuses_baseline_and_publishes_last(arm):
    cfg = configuration(arm)
    assert cfg.model.root._target_.endswith("TigerTrainingProbe")
    assert cfg.data.dataset_class._target_ == "src.data.datasets.SequenceDataset"
    assert cfg.model.root.statistics.max_num_sequences == 32
    assert cfg.model.root.statistics.training_data_dir == "data/beauty/training"
    assert cfg.model.root.statistics.source_split == "training"
    assert cfg.ckpt_path is None
    assert cfg.run_test_after_training is False
    assert cfg.trainer.root.max_steps == 2000
    assert cfg.model.training_model_config.optimizer.lr == 5e-5
    assert cfg.callbacks.wandb_checkpoint_writer.selection == "last"
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 2000
    assert cfg.callbacks.model_checkpoint.save_last is True
    assert {
        "encoder",
        "decoder",
        "semantic_ids",
        "num_hierarchies",
        "codebook_size",
        "embedding_dim",
        "training_model_config",
    } <= set(cfg.model.root)
    assert cfg.model.training_model_config._target_.endswith("TrainingModelConfig")
    assert cfg.model.training_model_config.optimizer._target_ == "torch.optim.Adam"
    assert cfg.model.metrics._target_.endswith("MetricEngine")
    assert cfg.callbacks.model_checkpoint._target_.endswith("ModelCheckpoint")
    assert cfg.callbacks.wandb_checkpoint_writer._target_.endswith("WandbCheckpointWriter")
    assert "wandb_artifact_lineage" in cfg.callbacks
    assert not {"root", "encoder", "decoder", "metrics", "model_checkpoint", "wandb_checkpoint_writer"} & set(cfg)


@pytest.mark.parametrize("arm", ["ce", "reweighted"])
def test_composed_model_and_callbacks_construct_with_tiny_local_inputs(tmp_path, monkeypatch, arm):
    cfg = configuration(
        arm,
        extra=[
            "codebook_size=4",
            "model.root.embedding_dim=4",
            "model.encoder.config.num_heads=2",
            "model.encoder.config.d_ff=8",
            "model.encoder.config.d_kv=2",
            "model.encoder.config.num_layers=1",
        ],
    )
    sids = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]])
    monkeypatch.setattr("src.data.components.artifacts.load_semantic_id_tensor", lambda **kwargs: sids.clone())
    monkeypatch.setattr(
        "src.data.components.tiger_training_statistics.load_training_statistics",
        lambda **kwargs: {
            "semantic_ids": sids.clone(),
            "expected_counts": torch.tensor([9.0, 1.0]),
            "metadata": {"source_split": "training"},
        },
    )
    original = Tiger(
        encoder=instantiate(cfg.model.encoder),
        decoder=instantiate(cfg.model.decoder),
        semantic_ids=sids,
        num_hierarchies=4,
        codebook_size=4,
        embedding_dim=4,
        training_model_config=instantiate(cfg.model.training_model_config),
        should_check_prefix=True,
    )
    checkpoint = tmp_path / "initial.ckpt"
    torch.save({"state_dict": original.state_dict()}, checkpoint)
    cfg.initialization_checkpoint_path = checkpoint.as_posix()
    model = instantiate(cfg.model.root)
    assert model.arm == arm
    assert isinstance(model.loss_function, torch.nn.CrossEntropyLoss)
    optimizer = model.optimizer(params=model.parameters())
    assert optimizer.param_groups[0]["lr"] == 5e-5
    assert set(model.state_dict()) == set(original.state_dict())
    cfg.paths.output_dir = tmp_path.as_posix()
    callback = instantiate(cfg.callbacks.model_checkpoint)
    assert callback.save_last and callback.save_top_k == 0
    assert instantiate(cfg.callbacks.wandb_checkpoint_writer).selection == "last"
    assert instantiate(cfg.callbacks.wandb_artifact_lineage) is not None


def test_statistics_follow_expansion_override_and_baseline_stays_intact():
    cfg = configuration(extra=["data.train_preprocessing_functions.4.max_num_sequences=16"])
    assert cfg.model.root.statistics.max_num_sequences == 16
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        original = compose(config_name="main", overrides=["experiment=tiger_train"])
    assert original.model.root._target_ == "src.recommendation.tiger.Tiger"
    assert original.model.training_model_config.optimizer.lr == 0.0005
    assert original.run_test_after_training is True


@pytest.fixture
def bash():
    git = shutil.which("git")
    candidates = [shutil.which("bash"), str(Path(git).parents[1] / "bin/bash.exe") if git else None]
    for candidate in candidates:
        if candidate and Path(candidate).exists() and Path(candidate).parent.name.lower() != "system32":
            check = subprocess.run([candidate, "-c", "exit 0"], capture_output=True)
            if check.returncode == 0:
                return candidate
    pytest.skip("Usable bash unavailable")


@pytest.fixture
def script(tmp_path):
    source = (ROOT / "tiger_training_probe.sh").read_text(encoding="utf-8")
    instrumented = source[: source.index("OMP_NUM_THREADS=")] + 'printf "%s\\n" "${ARGS[@]}"\n'
    path = tmp_path / "probe.sh"
    path.write_text(instrumented, encoding="utf-8", newline="\n")
    return path.as_posix()


def required_args():
    return [
        "--data-dir",
        "data/test set",
        "--semantic-id-path",
        "sid.pt",
        "--initialization-checkpoint-path",
        "a.ckpt",
        "--group",
        "rkmeans",
        "--devices",
        "[0]",
        "--arm",
        "ce",
        "--notes",
        'probe "paired", a=b',
    ]


def test_script_syntax(bash):
    result = subprocess.run([bash, "-n", (ROOT / "tiger_training_probe.sh").as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("equals", [False, True])
def test_script_quote_roundtrip_and_override_order(bash, script, equals):
    args = required_args()
    if equals:
        args = [f"{args[i]}={args[i + 1]}" for i in range(0, len(args), 2)]
    result = subprocess.run(
        [bash, script, *args, "--dry-run", "--seed=2024", "training_probe.alpha=0.1"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "training_probe.alpha=0.1"
    assert "--dry-run" in lines
    parsed = OverridesParser.create().parse_overrides([x for x in lines if x != "--dry-run"])
    values = {x.key_or_group: x.value() for x in parsed}
    assert values["logger.wandb.notes"] == 'probe "paired", a=b'
    assert values["data_dir"] == "data/test set"
    assert values["seed"] == 2024


@pytest.mark.parametrize(
    "extra",
    [
        ["--seed=-1"],
        ["--arm=bad"],
        ["--notes="],
        ["--data-dir", ""],
        ["--master-port=99999"],
        ["--unknown"],
        ["--devices"],
    ],
)
def test_script_invalid_inputs(bash, script, extra):
    result = subprocess.run([bash, script, *required_args(), *extra], capture_output=True, text=True)
    assert result.returncode == 2


def test_script_required_and_default_seed(bash, script):
    assert subprocess.run([bash, script], capture_output=True).returncode == 2
    result = subprocess.run([bash, script, *required_args()], capture_output=True, text=True)
    assert "seed=42" in result.stdout.splitlines()
