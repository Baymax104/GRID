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
