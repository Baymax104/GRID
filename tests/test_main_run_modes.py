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
