from omegaconf import OmegaConf

from src.common.analysis.runner import run_analysis_runner


class _Runnable:
    def __init__(self):
        self.ran = False

    def run(self):
        self.ran = True


class _NotRunnable:
    pass


def test_run_analysis_runner_instantiates_and_runs_configured_runner(monkeypatch):
    runner = _Runnable()
    cfg = OmegaConf.create({"analysis": {"runner": {"_target_": "tests.Dummy"}}})

    monkeypatch.setattr("src.common.analysis.runner.hydra.utils.instantiate", lambda _: runner)

    run_analysis_runner(cfg)

    assert runner.ran


def test_run_analysis_runner_rejects_missing_runner():
    cfg = OmegaConf.create({"analysis": {}})

    try:
        run_analysis_runner(cfg)
    except ValueError as exc:
        assert "cfg.analysis.runner" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing analysis runner.")


def test_run_analysis_runner_requires_callable_run(monkeypatch):
    cfg = OmegaConf.create({"analysis": {"runner": {"_target_": "tests.Dummy"}}})

    monkeypatch.setattr("src.common.analysis.runner.hydra.utils.instantiate", lambda _: _NotRunnable())

    try:
        run_analysis_runner(cfg)
    except TypeError as exc:
        assert "run()" in str(exc)
    else:
        raise AssertionError("Expected TypeError for non-runnable analysis runner.")
