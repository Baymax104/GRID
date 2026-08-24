from omegaconf import OmegaConf

import src.main as main_module


def test_run_dispatches_analysis_mode(monkeypatch):
    calls = []
    cfg = OmegaConf.create({"run_mode": "analysis"})

    monkeypatch.setattr(main_module, "run_analysis", lambda received: calls.append(received))

    main_module.run(cfg)

    assert calls == [cfg]


def test_run_error_mentions_analysis_mode():
    cfg = OmegaConf.create({"run_mode": "unknown"})

    try:
        main_module.run(cfg)
    except ValueError as exc:
        assert "train|inference|analysis" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported run_mode.")


def test_run_analysis_uses_lightning_test(monkeypatch):
    calls = []
    cfg = OmegaConf.create({"run_mode": "analysis"})

    class _Trainer:
        def test(self, model, datamodule, ckpt_path):
            calls.append(("test", model, datamodule, ckpt_path))

    class _PipelineModules:
        model = "model"
        datamodule = "datamodule"
        trainer = _Trainer()

    class _PipelineLauncher:
        def __enter__(self):
            calls.append(("enter",))
            return _PipelineModules()

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type))

    monkeypatch.setattr(main_module, "pipeline_launcher", lambda received: _PipelineLauncher())

    main_module.run_analysis(cfg)

    assert calls == [("enter",), ("test", "model", "datamodule", None), ("exit", None)]


def test_run_inference_resolves_checkpoint_path(monkeypatch):
    calls = []
    cfg = OmegaConf.create({"run_mode": "inference", "ckpt_path": "wandb://abc123", "project": "GRID"})

    class _Trainer:
        def predict(self, model, datamodule, ckpt_path, return_predictions):
            calls.append(("predict", ckpt_path, return_predictions))

    class _PipelineModules:
        def __init__(self, received_cfg):
            self.cfg = received_cfg
            self.model = "model"
            self.datamodule = "datamodule"
            self.trainer = _Trainer()

    class _PipelineLauncher:
        def __enter__(self):
            calls.append(("enter", cfg.ckpt_path))
            return _PipelineModules(cfg)

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type))

    def resolve_checkpoint_path(path, **kwargs):
        calls.append(("resolve_checkpoint_path", path, kwargs))
        return "resolved.ckpt"

    monkeypatch.setattr(main_module, "resolve_checkpoint_path", resolve_checkpoint_path)
    monkeypatch.setattr(main_module, "pipeline_launcher", lambda received: _PipelineLauncher())

    main_module.run_inference(cfg)

    assert calls == [
        ("resolve_checkpoint_path", "wandb://abc123", {"default_project": "GRID"}),
        ("enter", "resolved.ckpt"),
        ("predict", "resolved.ckpt", False),
        ("exit", None),
    ]
