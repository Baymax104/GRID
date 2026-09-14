import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.utils import instantiate

import src.utils.hydra_resolvers  # noqa: F401
from src.recommendation.tiger_catalog_grounded.module import ARMS

ROOT = Path(__file__).resolve().parents[1]


def configuration(arm="full", inference=False, extra=()):
    mode = "inference" if inference else "train"
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="main",
            overrides=[
                f"experiment=tiger_catalog_grounded_{mode}",
                "data_dir=data/beauty",
                "dataset_name=beauty",
                "devices=[0]",
                "group=rkmeans",
                "semantic_id_path=sid.pt",
                "embedding_path=embedding.pt",
                f"catalog_arm={arm}",
                *(["ckpt_path=checkpoint.ckpt", "data_split=evaluation"] if inference else []),
                *extra,
            ],
        )


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("inference", [False, True])
def test_all_arms_compose_and_instantiate(arm, inference, monkeypatch, tmp_path):
    cfg = configuration(
        arm,
        inference=inference,
        extra=[
            "num_hierarchies=2",
            "codebook_size=4",
            "beam_width=2",
            "model.root.embedding_dim=4",
            "model.root.projection_dim=3",
            "model.encoder.config.num_heads=2",
            "model.encoder.config.d_ff=8",
            "model.encoder.config.d_kv=2",
            "model.encoder.config.num_layers=1",
            *(["prefix_trace=false", "callbacks.prefix_trace_writer=null"] if inference and arm == "hybrid" else []),
        ],
    )
    sids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    values = dict(
        keys=torch.arange(4),
        semantic_ids=sids,
        embeddings=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, -1.0]]),
    )
    monkeypatch.setattr("src.data.components.artifacts.load_semantic_id_tensor", lambda **kw: sids)
    monkeypatch.setattr("src.data.components.catalog_content.load_catalog_content", lambda **kw: values)
    model = instantiate(cfg.model.root)
    assert model.arm == arm
    assert "wandb_artifact_lineage" in cfg.callbacks
    if inference:
        assert model.trace_prefix_survival == (arm != "hybrid")
        assert cfg.model.metrics is None
        return
    assert cfg.run_test_after_training is False and cfg.ckpt_path is None
    assert cfg.trainer.root.max_steps == 20000 and cfg.trainer.root.val_check_interval == 500
    assert cfg.model.training_model_config.optimizer.lr == 0.0005
    assert cfg.data.train_dataloader.batch_size_per_device == 128
    assert cfg.callbacks.model_checkpoint.monitor == "val/ndcg@10"
    assert cfg.callbacks.wandb_checkpoint_writer.selection == "best"
    assert "wandb_artifact_lineage" in cfg.callbacks
    cfg.paths.output_dir = tmp_path.as_posix()
    assert instantiate(cfg.callbacks.model_checkpoint).monitor == "val/ndcg@10"


def test_inference_config_preserves_target_identity_and_hybrid_writer_switch():
    cfg = configuration(inference=True)
    assert cfg.model.root.trace_prefix_survival
    assert cfg.data.collate.output_key_field_name == "user_id"
    assert cfg.model.metrics is None
    assert cfg.data.predict_dataloader.data_folder == "data/beauty/evaluation"
    hybrid = configuration("hybrid", inference=True, extra=["prefix_trace=false", "callbacks.prefix_trace_writer=null"])
    assert hybrid.callbacks.prefix_trace_writer is None
    assert hybrid.model.root.trace_prefix_survival is False


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
        "--semantic-id-path",
        "wandb://sid",
        "--embedding-path",
        "wandb://emb",
        "--dataset",
        "beauty",
        "--devices",
        "[0,1]",
        "--notes",
        'paired "run", x=y; $(literal)',
    ]
    if mode == "inference":
        args += ["--checkpoint-path", "wandb://checkpoint", "--data-split", "evaluation"]
    return args


def instrument(tmp_path, mode):
    source = (ROOT / f"tiger_catalog_grounded_{mode}.sh").read_text(encoding="utf-8")
    path = tmp_path / f"{mode}.sh"
    path.write_text(
        source[: source.index("OMP_NUM_THREADS=")] + 'printf "%s\\n" "${ARGS[@]}"\n', encoding="utf-8", newline="\n"
    )
    return path.as_posix()


@pytest.mark.parametrize("mode", ["train", "inference", "suite", "diagnosis"])
def test_script_syntax(bash, mode):
    result = subprocess.run(
        [bash, "-n", (ROOT / f"tiger_catalog_grounded_{mode}.sh").as_posix()], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("equals", [True, False])
def test_script_quotes_native_overrides_and_defaults(bash, tmp_path, mode, equals):
    args = required(mode)
    if equals:
        args = [f"{args[i]}={args[i + 1]}" for i in range(0, len(args), 2)]
    result = subprocess.run(
        [bash, instrument(tmp_path, mode), *args, "--dry-run", "--seed=17", "seed=31"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[-1] == "seed=31" and "seed=17" in lines and "--dry-run" in lines
    parsed = {
        item.key_or_group: item.value()
        for item in OverridesParser.create().parse_overrides([s for s in lines if s != "--dry-run"])
    }
    assert parsed["data_dir"] == "data/test set"
    assert parsed["logger.wandb.notes"] == 'paired "run", x=y; $(literal)'
    if mode == "inference":
        assert parsed["checkpoint_reference"] == "wandb://checkpoint"
    result = subprocess.run([bash, instrument(tmp_path, mode), *required(mode)], capture_output=True, text=True)
    assert result.returncode == 0 and "seed=42" in result.stdout and "--dry-run" not in result.stdout


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize(
    "bad",
    [["--data-dir="], ["--data-dir", " "], ["--seed=-1"], ["--master-port=0"], ["--arm=unknown"], ["--unknown=1"]],
)
def test_invalid_script_arguments_fail_before_launch(bash, tmp_path, mode, bad):
    result = subprocess.run([bash, instrument(tmp_path, mode), *required(mode), *bad], capture_output=True, text=True)
    assert result.returncode == 2 and "Error:" in result.stderr


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_missing_data_directory_fails(bash, tmp_path, mode):
    result = subprocess.run([bash, instrument(tmp_path, mode), *required(mode)[2:]], capture_output=True, text=True)
    assert result.returncode == 2


def test_hybrid_inference_disables_trace_before_native_overrides(bash, tmp_path):
    result = subprocess.run(
        [bash, instrument(tmp_path, "inference"), *required("inference"), "--arm=hybrid"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "prefix_trace=false" in result.stdout and "callbacks.prefix_trace_writer=null" in result.stdout


@pytest.mark.parametrize("queue", [1, 2])
def test_queues_run_six_correct_conditions_on_disjoint_gpus(bash, tmp_path, queue):
    shutil.copy(ROOT / "tiger_catalog_grounded_suite.sh", tmp_path)
    stub = tmp_path / "tiger_catalog_grounded_train.sh"
    stub.write_text(
        'printf "GPU=%s PROC=%s\\n" "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE"\nprintf "%s\\n" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    result = subprocess.run(
        [
            bash,
            "./tiger_catalog_grounded_suite.sh",
            f"--queue={queue}",
            "--beauty-data-dir=data/beauty",
            "--sports-data-dir=data/sports",
            "--beauty-semantic-id-path=beauty-sid",
            "--sports-semantic-id-path=sports-sid",
            "--beauty-embedding-path=beauty-emb",
            "--sports-embedding-path=sports-emb",
            "--dry-run",
            "--seed=17",
            "model.root.temperature=0.2",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    gpu = "0,1" if queue == 1 else "2,3"
    assert lines.count(f"GPU={gpu} PROC=2") == 6
    assert lines.count("--dry-run") == 6 and lines.count("model.root.temperature=0.2") == 6
    arms = [lines[i + 1] for i, line in enumerate(lines) if line == "--arm"]
    assert arms == (
        ["original", "full", "single_prototype", "shuffled", "original", "full"]
        if queue == 1
        else ["mask_ce", "hybrid", "token_content_init", "no_aux", "mask_ce", "hybrid"]
    )
    datasets = [lines[i + 1] for i, line in enumerate(lines) if line == "--data-dir"]
    assert datasets == ["data/beauty"] * 4 + ["data/sports"] * 2


def test_diagnosis_composition_and_explicit_split_without_trace():
    from types import SimpleNamespace

    from src.data.catalog_diagnosis import CatalogDiagnosisDataset

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=tiger_catalog_grounded_diagnosis",
                "data_dir=data/beauty",
                "data_split=evaluation",
                "semantic_id_path=sid.pt",
                "raw_num_hierarchies=3",
                "group=rkmeans",
            ],
        )
    assert cfg.run_mode == "analysis"
    assert cfg.data.test_dataloader.dataset_class._target_ == "src.data.catalog_diagnosis.CatalogDiagnosisDataset"
    factory = instantiate(cfg.data.test_dataloader.dataset_class)
    dataset = factory(dataset_config=None, data_folder="unused", semantic_id_path="unused", raw_num_hierarchies=3)
    assert isinstance(dataset, CatalogDiagnosisDataset) and dataset._trace_data_split() == "evaluation"
    dataset.fixed_prefix_trace = SimpleNamespace(metadata={"data_split": "testing"})
    with pytest.raises(ValueError, match="disagrees"):
        dataset._trace_data_split()


def test_diagnosis_wrapper_preserves_notes_dry_run_and_native_overrides(bash, tmp_path):
    shutil.copy(ROOT / "tiger_catalog_grounded_diagnosis.sh", tmp_path)
    source = (ROOT / "tail_sid_diagnosis.sh").read_text(encoding="utf-8")
    (tmp_path / "tail_sid_diagnosis.sh").write_text(
        source[: source.index("uv run --module src.main")] + 'printf "%s\\n" "${ARGS[@]}"\n',
        encoding="utf-8",
        newline="\n",
    )
    result = subprocess.run(
        [
            bash,
            "./tiger_catalog_grounded_diagnosis.sh",
            "--data-dir=data/beauty",
            "--semantic-id-path=wandb://sid",
            "--group=rkmeans",
            "--data-split=evaluation",
            "--embedding-path=wandb://embedding",
            '--notes=paired "run", x=y',
            "--dry-run",
            "seed=17",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    values = {
        item.key_or_group: item.value()
        for item in OverridesParser.create().parse_overrides([s for s in lines if s != "--dry-run"])
    }
    assert values["experiment"] == "tiger_catalog_grounded_diagnosis"
    assert values["data_split"] == "evaluation" and values["embedding_path"] == "wandb://embedding"
    assert values["logger.wandb.notes"] == 'paired "run", x=y'
    assert values["seed"] == 17 and "--dry-run" in lines
